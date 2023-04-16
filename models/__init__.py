from .cgcnn import CGCNN
from .crystal_transformer import CrystalTransformer
from .sat import GraphTransformer


ALL_MODELS = {
    "cgcnn": CGCNN,
    "cryspher": None,
    "crystal_transformer": CrystalTransformer,
    "graph_transformer": GraphTransformer,
}
