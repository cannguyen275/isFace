import torch
import onnx
from backbone.shufflenet_v2 import shufflenet_v2_x0_5
from helper.utils import load_checkpoint

if __name__ == '__main__':
    model = shufflenet_v2_x0_5()
    model = load_checkpoint(model,
                            '/home/can/AI_Camera/face_clasification/model/checkpoint_149_0.010453527558476049.tar')
    model.eval()
    # ##################export###############
    output_onnx = "model_filter.onnx"
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, 112, 112)
    torch_out = torch.onnx._export(model,
                                   inputs,
                                   output_onnx,
                                   verbose=True,
                                   input_names=input_names,
                                   output_names=output_names,
                                   example_outputs=True,  # to show sample output dimension
                                   keep_initializers_as_inputs=True,  # to avoid error _Map_base::at
                                   opset_version=7, # need to change to 11, to deal with tensorflow fix_size input
                                   # dynamic_axes={
                                   #     "input0": [2, 3],
                                   #     "loc0": [1, 2],
                                   #     "conf0": [1, 2],
                                   #     "landmark0": [1, 2]
                                   # }
                                   )

    onnx_model = onnx.load(output_onnx)

    print('The model is:\n{}'.format(onnx_model))

    # Check the model
    onnx.checker.check_model(onnx_model)
    print('The model is checked!')

    ##################end###############
# snpe_onnx_dlc -i input_model -out_put outmodel
