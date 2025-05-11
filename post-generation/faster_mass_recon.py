import os
from multiprocessing import Pool
import subprocess

org_data_dir = '../data/generated_nprint'
out_put_data_dir = '../data/replayable_generated_pcaps'
out_put_nprint_dir = '../data/replayable_generated_nprints'

def process_nprint(org_nprint):
    org_nprint_path = os.path.join(org_data_dir, org_nprint)
    output_pcap_path = os.path.join(out_put_data_dir, org_nprint.replace('.nprint', '.pcap'))
    output_nprint_path = os.path.join(out_put_nprint_dir, org_nprint)

    cmd = [
        "python3", "reconstruction.py",
        "--generated_nprint_path", org_nprint_path,
        "--formatted_nprint_path", "correct_format.nprint",
        "--output", output_pcap_path,
        "--nprint", output_nprint_path
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Success: {org_nprint}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {org_nprint} - {e.stderr}")
        return False

if __name__ == '__main__':
    nprint_files = [f for f in os.listdir(org_data_dir) if f.endswith('.nprint')]
    num_processes = min(os.cpu_count(), len(nprint_files)) or 1
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_nprint, nprint_files)
    print(f"Processed {sum(results)}/{len(nprint_files)} files successfully")
