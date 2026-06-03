import h5py
import json
import sys
import os

def verify_ilp(file_path):
    print(f"[*] Starting verification of: {file_path}")
    if not os.path.exists(file_path):
        print(f"[-] ERROR: File not found: {file_path}")
        return False

    success = True

    def log_success(msg):
        print(f"[+] {msg}")

    def log_error(msg):
        nonlocal success
        success = False
        print(f"[-] ERROR: {msg}")

    def log_warn(msg):
        print(f"[!] WARNING: {msg}")

    try:
        with h5py.File(file_path, 'r') as f:
            # --- Root Level ---
            print("\n[ Root Level ]")
            if "ilastikVersion" in f:
                ver = f["ilastikVersion"][()]
                log_success(f"ilastikVersion: {ver.decode('utf-8') if isinstance(ver, bytes) else ver}")
            else:
                log_error("Missing 'ilastikVersion'")

            if "workflowName" in f:
                wf = f["workflowName"][()]
                wf_str = wf.decode('utf-8') if isinstance(wf, bytes) else wf
                if wf_str == "Pixel Classification":
                    log_success(f"workflowName: {wf_str}")
                else:
                    log_error(f"Incorrect workflowName: Expected 'Pixel Classification', got '{wf_str}'")
            else:
                log_error("Missing 'workflowName'")

            if "time" in f:
                log_success(f"time: {f['time'][()]}")
            else:
                log_warn("Missing 'time' dataset")

            # --- Input Data ---
            print("\n[ Input Data ]")
            if "Input Data" in f:
                id_group = f["Input Data"]
                if "StorageVersion" in id_group:
                    sv = id_group["StorageVersion"][()]
                    log_success(f"StorageVersion: {sv.decode('utf-8') if isinstance(sv, bytes) else sv}")
                else:
                    log_error("Input Data/StorageVersion missing")

                if "infos" in id_group:
                    infos = id_group["infos"]
                    log_success(f"Found {len(infos)} lanes in infos")
                    for lane_name in infos:
                        lane = infos[lane_name]
                        if "Raw Data" in lane:
                            raw = lane["Raw Data"]
                            log_success(f"  {lane_name}/Raw Data found")
                            # Check required fields in Raw Data
                            for field in ["filePath", "nickname", "axistags", "shape"]:
                                if field in raw:
                                    val = raw[field][()]
                                    if field == "axistags":
                                        try:
                                            tags = json.loads(val.decode('utf-8') if isinstance(val, bytes) else val)
                                            if "axes" in tags:
                                                log_success(f"    {field}: Valid JSON")
                                            else:
                                                log_error(f"    {field}: Invalid structure (missing 'axes')")
                                        except Exception as e:
                                            log_error(f"    {field}: Failed to parse JSON: {e}")
                                    else:
                                        log_success(f"    {field}: Present")
                                else:
                                    log_error(f"    {field}: MISSING in Raw Data")
                        else:
                            log_error(f"  {lane_name} missing 'Raw Data' group")
                else:
                    log_error("Input Data/infos missing")
            else:
                log_error("Missing 'Input Data' group")

            # --- Feature Selections ---
            print("\n[ Feature Selections ]")
            if "FeatureSelections" in f:
                fs = f["FeatureSelections"]
                for field in ["FeatureIds", "Scales", "SelectionMatrix"]:
                    if field in fs:
                        log_success(f"{field}: Present")
                    else:
                        log_error(f"FeatureSelections/{field} missing")
                
                if "SelectionMatrix" in fs and "FeatureIds" in fs and "Scales" in fs:
                    matrix_shape = fs["SelectionMatrix"].shape
                    expected_shape = (len(fs["FeatureIds"]), len(fs["Scales"]))
                    if matrix_shape == expected_shape:
                        log_success(f"SelectionMatrix shape matches: {matrix_shape}")
                    else:
                        log_error(f"SelectionMatrix shape mismatch: Expected {expected_shape}, got {matrix_shape}")
            else:
                log_error("Missing 'FeatureSelections' group")

            # --- Pixel Classification ---
            print("\n[ Pixel Classification ]")
            if "PixelClassification" in f:
                pc = f["PixelClassification"]
                for field in ["LabelNames", "LabelColors", "LabelSets"]:
                    if field in pc:
                        log_success(f"{field}: Present")
                    else:
                        log_error(f"PixelClassification/{field} missing")
                
                if "LabelSets" in pc:
                    ls = pc["LabelSets"]
                    log_success(f"Found {len(ls)} label sets")
                    for label_set_name in ls:
                        lset = ls[label_set_name]
                        log_success(f"  {label_set_name} found")
                        for block_name in lset:
                            block = lset[block_name]
                            log_success(f"    {block_name} found")
                            if "axistags" in block.attrs:
                                try:
                                    json.loads(block.attrs["axistags"])
                                    log_success("      axistags attr: Valid JSON")
                                except:
                                    log_error("      axistags attr: Invalid JSON")
                            else:
                                log_error("      axistags attr: MISSING")
                            
                            if "blockSlice" in block.attrs:
                                bs = block.attrs["blockSlice"]
                                bs_str = bs.decode('utf-8') if isinstance(bs, bytes) else bs
                                if bs_str.startswith("[") and bs_str.endswith("]") and ":" in bs_str:
                                    log_success(f"      blockSlice attr: {bs_str}")
                                else:
                                    log_error(f"      blockSlice attr: Invalid format '{bs_str}'")
                            else:
                                log_error("      blockSlice attr: MISSING")

                if "ClassifierFactory" in pc:
                    log_success("ClassifierFactory: Present")
                else:
                    log_warn("ClassifierFactory: Missing (Ilastik will use defaults)")
                
                if "ClassifierForests" in pc:
                    log_success("ClassifierForests: Present (Trained)")
                else:
                    log_warn("ClassifierForests: Missing (Ilastik will require retraining)")
            else:
                log_error("Missing 'PixelClassification' group")

    except Exception as e:
        log_error(f"Unexpected error during verification: {e}")
        return False

    print("\n" + "="*40)
    if success:
        print("[***] VERIFICATION SUCCESSFUL: Project meets the spec.")
    else:
        print("[!!!] VERIFICATION FAILED: Project has errors.")
    print("="*40)
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ILP_PixelClassification_verify.py <path_to_project.ilp>")
        sys.exit(1)
    
    verify_ilp(sys.argv[1])
