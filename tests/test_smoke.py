import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import fit_and_evaluate;from src.core import fit_distribution_mle,var
def test_data():d=make_synthetic(500);assert d["n_samples"]==500
def test_dist_fit():assert fit_distribution_mle([100,200,300,400,500],"gamma")["ks_stat"]>=0
def test_fit():d=make_synthetic(500);m,met=fit_and_evaluate(d);assert"best_dist"in met
if __name__=="__main__":test_data();test_dist_fit();test_fit();print("OK")
