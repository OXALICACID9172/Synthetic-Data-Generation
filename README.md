## Installation (Current support for Linux only)

```bash
# python version requirement 3.10
# SSH to Linux server via designated port (see following for example)
ssh -L 7860:LocalHost:7860 username@server_address

# Clone the repository
git clone --recurse-submodules https://github.com/noise-lab/NetDiffusion_Generator.git

# Create and activate the conda env
conda create -n NetDiffusion python=3.10
conda activate NetDiffusion

# Navigate to the project directory
cd NetDiffusion_Generator

# Install dependencies in the virtual env of choice (we recommend pyenv, if using conda install using conda_requirements.txt)
pip install -r requirements.txt

# Install `nprint` from https://nprint.github.io/nprint/install.html
```

## Import Data
Store raw pcaps used for fine-tuning into 'NetDiffusion_Code/data/fine_tune_pcaps' with the application/service labels as the filenames, e.g.,'netflix_01.pcap'.

## Data Preprocessing
```bash
# Navigate to preprocessing dir
cd data_preprocessing/

# Run preprocessing conversions
python3 pcap_to_img.py

# Navigate to fine-tune dir and the khoya subdir for task creation
# (replace the number in 20_network with the average number of pcaps per traffic type used for fine-tuning)
cd ../fine_tune/kohya_ss_fork/model_training/
mkdir -p example_task/{image/20_network,log,model}

# Leverage Stable Diffusion WebUI for initial caption creation
cd ../../sd-webui-fork/stable-diffusion-webui/
# Lunch WebUI
bash webui.sh
```
1. Open the WebUI via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Under `extras/batch_from_directory`, enter the absolute path for `/NetDiffusion_Code/data/preprocessed_fine_tune_imgs` and `/NetDiffusion_Code/fine_tune/kohya_ss_fork/model_training/test_task/image/20_network` as the input/output directories.
2. Under `extras/batch_from_directory`, set the `scale to` parameter to `width = 816` and `height = 768` for resource-friendly fine-tuning (adjust based on resource availability).
3. Enable the `caption` parameter under `extras/batch_from_directory` and click `generate`.
4. Terminate `webui.sh`

```bash
# Change the caption prompt for explicit prompt-to-image correlation,
# For example, 'pixelated network data, type-0' refers to NetFlix pcap,
# Adjust the script based on fine-tuning task.
cd ../../ && python3 caption_changing.py
```

## Fine-Tuning
```bash
# Navigate to fine-tuning directory
cd kohya_ss_fork
# Grant execution access
chmod +x ./setup.sh
# Set up configuration
./setup.sh
# Set up accelerate environment (gpu and fp16 recommended)
accelerate config
# Fine-tune interface initialization
bash gui.sh
```
1. Open the fine-tuning interface via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Under `LoRA\Training`, load the configuration file via the absolute path for '/NetDiffusion_Code/fine_tune/LoraLowVRAMSettings.json'
3. Under `LoRA\Training\Folders`, enter the absolute paths for `/NetDiffusion_Code/fine_tune/kohya_ss_fork/model_training/test_task/image`,`/NetDiffusion_Code/fine_tune/kohya_ss_fork/model_training/test_task/model`, and `/NetDiffusion_Code/fine_tune/kohya_ss_fork/model_training/test_task/log` for the Image/Output/Logging folders, respectively, and adjust the model name if needed.
4. Under `LoRA\Training\Parameters\Basic`, adjust the Max Resolution to match the resolution from data preprocessing, e.g., 816,768.
5. Click on Start Training to begin the fine-tuning. Adjust the fine-tuning parameters as needed due to different generation tasks may have different parameter requirement to yield better synthetic data quality.

## Cuda Compatability Issues
1. If you are facing issues related to cuda I recommend following the steps in the link provided https://www.reddit.com/r/StableDiffusion/comments/173r5g0/kohya_error_libcudartso_is_not_in_the_ld_library/

## Generation
```bash
# Copy the fine-tuned LoRA model (adjust path namings as needed) to Stable Diffusion WebUI
cp model_training/test_task/model/test_task_model.safetensors ../sd-webui-fork/stable-diffusion-webui/models/Lora/
# Navigate to the generation directory
cd ../sd-webui-fork/stable-diffusion-webui/
# Initialize Stable Diffusion WebUI
bash webui.sh
```
1. Open the WebUI via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Install ControlNet extension for the WebUI and restart the WebUI: https://github.com/Mikubill/sd-webui-controlnet
3. To generate an image representation of a network trace, enter the corresponding caption prompt with the LoRA model extension under 'txt2img'. For example 'pixelated network data, type-0 \<lora:test_task_model:1\>' for NetFlix data.
4. Adjust the generation resolution to match the resolution from data preprocessing, e.g., 816,768.
5. Adjust the seed to match the seed used in fine-tuning, default is `1234`.
6. Enable Hires.fix to scale to `1088, 1024`.
7. From training data, sample a real pcap image (that belongs to the same category as the desired synthetic traffic) as input to the ControlNet interface, and set the Control Type (we recommend canny).
8. Click `Generate` to complete the generation.
Note that extensive adjustments on the generation and ControlNet parameters may be needed to yield the best generation result as the generation tasks and training data differ from each other.

## Post-Generation Heuristic Correction
1. Once enough instances of image representations of desired synthetic traffic are generated, place all of such instances under `/NetDiffusion_Code/data/generated_imgs`.
2. Navigate to the `/NetDiffusion_Code/post-generation/` folder
```bash
# Run the following to do the post-generation heuristic for reconversion back to pcaps and protocol compliance checking.
# Adjust the color processing threshold in color_processor.py as required for best generation results.
python3 color_processor.py && python3 img_to_nprint.py && python3 mass_reconstruction.py
```
This completes the post-generation pipeline with the final nprints and pcaps stored in `/NetDiffusion_Code/data/replayable_generated_nprints` and `/NetDiffusion_Code/data/replayable_generated_pcaps`, respectively.

## Evaluation
1. Open the evaluation Folder.
2. Run new_eval.py with the appropriate paths set.

## Citing NetDiffusion
```
@article{jiang2024netdiffusion,
  title={NetDiffusion: Network Data Augmentation Through Protocol-Constrained Traffic Generation},
  author={Jiang, Xi and Liu, Shinan and Gember-Jacobson, Aaron and Bhagoji, Arjun Nitin and Schmitt, Paul and Bronzino, Francesco and Feamster, Nick},
  journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
  volume={8},
  number={1},
  pages={1--32},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
