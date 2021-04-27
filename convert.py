import onnx
from onnx_tf.backend import prepare
import torch.onnx
from torch.autograd import Variable

from skylar.utils import load_model_from_state_dict
from skylar.utils import DeeperDeepSEA
from skylar.utils import NonStrandSpecific

if True:
    root, file = "analyzing/", "example_deeperdeepsea"
    mediator = root + "mediator.onnx"
    a = torch.load(root + file + ".pth.tar", map_location=lambda storage, location: storage)
    x = load_model_from_state_dict(a, NonStrandSpecific(DeeperDeepSEA(1000, 919)))
    # noinspection PyTypeChecker,PyArgumentList
    torch.onnx.export(x, Variable(torch.Tensor(320, 4, 8)), mediator)
    # noinspection PyUnresolvedReferences
    prepare(onnx.load(mediator)).export_graph(root + file + ".pb")
