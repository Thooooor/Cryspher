from data import CrystalBase, FULL_DATASETS
from time import time

DATASET_DICT = {
    "band_gap": "matbench_mp_gap.json",
    "e_form": "matbench_mp_e_form.json",
    "phonons": "matbench_phonons.json",
    "perovskites": "matbench_perovskites.json",
    "log_gvrh": "matbench_log_gvrh.json",
    "log_kvrh": "matbench_log_kvrh.json",
    "dielectric": "matbench_dielectric.json",
    "jdft2d": "matbench_jdft2d.json",
}


data_dir = "./datasets/e_form/"

def main():
    print("processing...")
    for dataset in FULL_DATASETS:
        print(dataset)
        start = time()
        data_dir = f"./datasets/{dataset}/"
        data_file = DATASET_DICT[dataset]
        CrystalBase(dataset, data_dir, data_file)
        print(f"Time: {time() - start}")
    print("done.")


if __name__ == "__main__":
    main()
