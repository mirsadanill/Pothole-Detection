import math
import os
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time


TRT_LOGGER = trt.Logger()


# model_h = 544
# model_w = 960
# stride = 16
# box_norm = 35.0

# grid_h = int(model_h / stride)
# grid_w = int(model_w / stride)
# grid_size = grid_h * grid_w

# grid_centers_w = []
# grid_centers_h = []

# for i in range(grid_h):
#     value = (i * stride + 0.5) / box_norm
#     grid_centers_h.append(value)

# for i in range(grid_w):
#     value = (i * stride + 0.5) / box_norm
#     grid_centers_w.append(value)

class tensor_engine:
    def __init__(self, engine_file, class_names_file, detection_save_path, image_size_h_w=[576, 768], batch_size=1):
        self.numberofclasses = 1
        self.buffers = []
        self.conf_thresh = 0.2
        self.nms_thresh = 0.3
        self.batch_size = batch_size
        self.class_names_dir = class_names_file
        self.class_names = []
        self.ctx = None
        self.boxes = None
        self.name = ""
        self.engine = None
        self.engine_path = engine_file
        self.engine_input_size = [batch_size, 3, image_size_h_w[0], image_size_h_w[1]]
        self.trt_outputs = []
        self.width = None
        self.height = None
        self.image = None
        self.imagecv2 = None
        self.save_path = detection_save_path
        self.i_start = float()
        self.i_started_cv2 = float()
        self.i_image_m = float()
        self.i_end = float()
        self.clear()
        self.load_class_names()
        self.create_engine()

    def clear(self):
        self.boxes = None
        self.name = ""
        self.width = None
        self.height = None
        self.image = None
        self.imagecv2 = None
        self.trt_outputs = []

    def load_class_names(self):
        with open(self.class_names_dir, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            self.class_names.append(line)
        self.numberofclasses = len(self.class_names)

    def create_engine(self):
        cuda.init()
        self.ctx = cuda.Device(0).make_context()
        self.load_engine()
        self.engine.create_execution_context().set_binding_shape(0, tuple(self.engine_input_size))
        self.allocate_buffers()


    def load_engine(self):
        print("Reading engine from file {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        """Allocates host and device buffer for TRT engine inference.
        This function is similair to the one in common.py, but
        converts network outputs (which are np.float32) appropriately
        before writing them to Python buffer. This is needed, since
        TensorRT plugins doesn't support output type description, and
        in our particular case, we use NMS plugin as network output.
        Args:
            engine (trt.ICudaEngine): TensorRT engine
        Returns:
            inputs [HostDeviceMem]: engine input memory
            outputs [HostDeviceMem]: engine output memory
            bindings [int]: buffer to device bindings
            stream (cuda.Stream): cuda stream for engine inference synchronization
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        binding_to_type = {
            'Input': np.float32,
            'BatchedNMS': np.int32,
            'BatchedNMS_1': np.float32,
            'BatchedNMS_2': np.float32,
            'BatchedNMS_3': np.float32
            }
        # Current NMS implementation in TRT only supports DataType.FLOAT but
        # it may change in the future, which could brake this sample here
        # when using lower precision [e.g. NMS output would not be np.float32
        # anymore, even though this is assumed in binding_to_type]

        for binding in self.engine:
            
            dims = self.engine.get_binding_shape(binding)
            size = trt.volume(dims) * self.batch_size
            
            # in case batch dimension is -1 (dynamic)
            if dims[0] < 0:
                size *= -1

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            #dtype = binding_to_type[str(binding)]

            # Allocate host and device buffers
            # host_mem = cuda.pagelocked_empty(size, dtype)
            host_mem = cuda.pagelocked_empty(size, np.float32)

            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append([host_mem, device_mem])
            else:
                outputs.append([host_mem, device_mem])
        # self.engine.create_execution_context().execute_v2(bindings)
        self.buffers = [inputs, outputs, bindings, stream]

    # Called for each inference request #------------------------------------------
    def read_frame(self, image, name):
        self.i_start = time.time()
        self.clear()
        self.name = name
        self.imagecv2 = image
        # self.imagecv2 = cv2.resize(self.imagecv2, (self.engine_input_size[3], self.engine_input_size[2]))
        # self.engine_input_size = [1, 3, self.imagecv2.shape[0], self.imagecv2.shape[1]]
        self.i_started_cv2 = time.time()
        self.detect()

    def detect(self):
        self.resized_imagecv2 = cv2.resize(self.imagecv2, (self.engine_input_size[3], self.engine_input_size[2]), interpolation=cv2.INTER_LINEAR)
        self.resized_imagecv2_in = cv2.cvtColor(self.resized_imagecv2, cv2.COLOR_BGR2RGB)
        self.resized_imagecv2_in = np.transpose(self.resized_imagecv2_in, (2, 0, 1)).astype(np.float32)
        self.resized_imagecv2_in = np.expand_dims(self.resized_imagecv2_in, axis=0)
        self.resized_imagecv2_in /= 255.0
        self.resized_imagecv2_in = np.ascontiguousarray(self.resized_imagecv2_in)
        self.buffers[0][0][0] = self.resized_imagecv2_in
        self.i_image_m = time.time()
        self.inference()


        # self.buffers = [inputs, outputs, bindings, stream].
        # for each buffer: 1 for device, 0 for host
    def inference(self):
        tt1 = time.time()
        self.ctx.push()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp[1], inp[0], self.buffers[3]) for inp in self.buffers[0]]

        tt2 = time.time()
        # Run inference.
        self.engine.create_execution_context().execute_async(bindings=self.buffers[2], batch_size=self.batch_size, stream_handle=self.buffers[3].handle)

        tt3 = time.time()
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out[0], out[1], self.buffers[3])for out in self.buffers[1]]

        tt4 = time.time()
        # Synchronize the stream
        self.buffers[3].synchronize()

        tt5 = time.time()
        # Return only the host outputs.
        self.trt_outputs = [out[0] for out in self.buffers[1]]

        tt6 = time.time()
        self.trt_outputs[0] = self.trt_outputs[0].reshape(1, -1, 1, 4)
        self.trt_outputs[1] = self.trt_outputs[1].reshape(1, -1, self.numberofclasses)

        self.i_end = time.time()
        self.ctx.pop()
        print('-----------------------------------')
        print('    TRT inference time: %f' % (self.i_end - self.i_start))
        print('-----------------------------------')
        print('              cv2 time: %f' % (-self.i_start + self.i_started_cv2))
        print('    image process time: %f' % (self.i_image_m - self.i_started_cv2))
        print(' actual inference time: %f' % (self.i_end - self.i_image_m))
        print('-----------------------------------')
        print('           to GPU time: %f' % (tt2 - tt1))
        print('        inference time: %f' % (tt3 - tt2))
        print('         from GPU time: %f' % (tt4 - tt3))
        print('      synchronize time: %f' % (tt5 - tt4))
        print('        returning time: %f' % (tt6 - tt5))
        print('-----------------------------------')

        self.post_processing()
        # self.plot_boxes_cv2()
    def post_processing_new(self,wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        bbs = []
        class_ids = []
        scores = []
        for c in range(self.numberofclasses):

            x1_idx = c * 4 * grid_size
            y1_idx = x1_idx + grid_size
            x2_idx = y1_idx + grid_size
            y2_idx = x2_idx + grid_size

            boxes = self.trt_outputs[0]
            for h in range(grid_h):
                for w in range(grid_w):
                    i = w + h * grid_w
                    score = self.trt_outputs[1][c * grid_size + i]
                    if score.any() >= self.conf_thresh:
                        o1 = boxes[x1_idx + w + h * grid_w]
                        o2 = boxes[y1_idx + w + h * grid_w]
                        o3 = boxes[x2_idx + w + h * grid_w]
                        o4 = boxes[y2_idx + w + h * grid_w]

                        o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)

                        xmin = int(o1)
                        ymin = int(o2)
                        xmax = int(o3)
                        ymax = int(o4)
                        if wh_format:
                            bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        else:
                            bbs.append([xmin, ymin, xmax, ymax])
                        class_ids.append(c)
                        scores.append(float(score))
        indexes = cv2.dnn.NMSBoxes(bbs, scores, self.nms_thresh, self.nms_thresh)
        for idx in indexes:
            idx = int(idx)
            xmin, ymin, w, h = bbs[idx]
            class_id = class_ids[idx]
            color = [255, 0, 0] if class_id else [0, 0, 255]
            cv2.rectangle(self.imagecv2, (xmin, ymin), (xmin + w, ymin + h), color, 2)
        #return bbs, class_ids, scores

    def applyBoxNorm(o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - grid_centers_w[x]) * -box_norm
        o2 = (o2 - grid_centers_h[y]) * -box_norm
        o3 = (o3 + grid_centers_w[x]) * box_norm
        o4 = (o4 + grid_centers_h[y]) * box_norm
        return o1, o2, o3, o4

    def post_processing(self):

        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # num_anchors = 9
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

        # [batch, num, 1, 4]
        box_array = self.trt_outputs[0]
        # [batch, num, num_classes]
        confs = self.trt_outputs[1]

        t1 = time.time()

        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        t2 = time.time()

        bboxes_batch = []
        for i in range(box_array.shape[0]):

            argwhere = max_conf[i] > self.conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = self.nms_cpu(ll_box_array, ll_max_conf, self.nms_thresh)

                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                      ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

            bboxes_batch.append(bboxes)

        t3 = time.time()

        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')

        self.boxes = bboxes_batch

    def nms_cpu(self, boxes, confs, min_mode=False):
        # print(boxes.shape)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= self.nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

    # Called for each inference request #------------------------------------------
    def save_frame_label(self):
        if len(self.boxes[0]) >= 1:
            cv2.imwrite(os.path.join(self.save_path, f"images/{self.name}.jpg"), self.imagecv2)
            self.plot_boxes_cv2()
            with open(os.path.join(self.save_path, f"labels/{self.name}.txt"), "w") as f:
                for i in range(len(self.boxes[0])):
                    box = self.boxes[0][i]
                    f.write(f"{str(box[6])} {str((box[0]+box[2])/2.0)} {str((box[1]+box[3])/2.0)} {str(box[2]-box[0])} {str(box[3]-box[1])}\n")
            cv2.imwrite(os.path.join(self.save_path, f"boxed/{self.name}.jpg"), self.imagecv2)


    def plot_boxes_cv2(self):

        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [
                          1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        self.height, self.width = self.imagecv2.shape[:2]

        if len(self.boxes[0]) >= 1:
            for i in range(len(self.boxes[0])):
                box = self.boxes[0][i]
                x1 = int(box[0] * self.width)
                y1 = int(box[1] * self.height)
                x2 = int(box[2] * self.width)
                y2 = int(box[3] * self.height)
                bbox_thick = int(0.6 * (self.height + self.width) / 600)

                rgb = None
                if len(box) >= 7 and self.class_names:
                    cls_conf = box[5]
                    cls_id = box[6]
                    print('%s: %f' % (self.class_names[cls_id], cls_conf))
                    classes = len(self.class_names)
                    offset = cls_id * 123457 % classes
                    red = get_color(2, offset, classes)
                    green = get_color(1, offset, classes)
                    blue = get_color(0, offset, classes)

                    rgb = (red, green, blue)

                    msg = str(self.class_names[cls_id]) + \
                        " "+str(round(cls_conf, 3))
                    t_size = cv2.getTextSize(
                        msg, 0, 0.7, thickness=bbox_thick // 2)[0]
                    c1, c2 = (x1, y1), (x2, y2)
                    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                    cv2.rectangle(self.imagecv2, (x1, y1), (np.float32(
                        c3[0]), np.float32(c3[1])), rgb, -1)
                    self.imagecv2 = cv2.putText(self.imagecv2, msg, (c1[0], np.float32(
                        c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

                self.imagecv2 = cv2.rectangle(
                    self.imagecv2, (x1, y1), (x2, y2), rgb, bbox_thick)

    



