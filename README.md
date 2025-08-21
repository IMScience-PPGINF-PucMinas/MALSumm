# Streamlined Extended Long Short-Term Memory for Video Skimming
PyTorch code for our Sibgrapi 2025 paper "Memory-Augmented Long Short-Term Memory for Dynamic Video Summarization" Enhanced
by [Leonardo Vilela Cardoso](http://lattes.cnpq.br/6741312586742178), [Barbara Hellen P. Soraggi](http://lattes.cnpq.br/9625828407766581), [Silvio Jamil F. Guimarães](http://lattes.cnpq.br/8522089151904453) and 
[Zenilton K. G. Patrocínio Jr](http://lattes.cnpq.br/8895634496108399), 

Capturing relevant content from videos while preserving temporal coherence remains a central challenge in video skimming. The prevalence of redundant information often hinders the extraction of meaningful content, especially when the goal is to retain the central narrative of the video. While scene change detection can aid in segmenting video content, conventional methods often struggle with highly diverse and repetitive scenes due to their limited ability to model temporal dependencies and detect transitions effectively. To address these limitations, we propose the \methodNamefull\ -- \methodName, a supervised architecture based on Extended Long Short-Term Memory (xLSTM) networks that enhances memory capacity through a dual-path design. This design integrates weighted memory to evaluate local and global information, allowing the model to preserve fine-grained details while maintaining overall temporal consistency, all within a low-complexity framework. Experimental results validate the effectiveness of our approach, achieving an average F-score of 49.7 on the SumMe dataset and 62.1 on TVSum, outperforming recent supervised baselines. Additionally, when measuring alignment with human annotations, the model attains a Kendall's $\tau$ of 0.180 and Spearman's $\rho$ of 0.242, exceeding the scores reported for human agreement. These findings demonstrate that our method provides a competitive and lightweight solution for dynamic video summarization, effectively balancing accuracy and efficiency.

## Main dependencies
Developed, checked and verified on an `Ubuntu 22.04` PC with a `GTX 1080 SUPER` GPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.9 | 2.4.1 | 11.0 | 8005 | 2.4.1 | 2.3.0 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
</pre>
Original videos and annotations for each dataset are also available in the dataset providers' webpages: 
- <a href="https://github.com/yalesong/tvsum"><img src="https://img.shields.io/badge/Dataset-TVSum-green"/></a> <a href="https://gyglim.github.io/me/vsum/index.html#benchmark"><img src="https://img.shields.io/badge/Dataset-SumMe-blue"/></a>
</div>



## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](model/data_loader.py), specify the path to the h5 file of the used dataset, and the path to the JSON file containing data about the utilized data splits.
 - In [`configs.py`](model/configs.py), define the directory where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'SumMe' | 'SumMe', 'TVSum'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--seed` | Chosen number for generating reproducible random numbers. | 12345 | None, int
`--fusion` | Type of the used approach for feature fusion. | 'add' | None, 'add', 'mult', 'avg', 'max' 
`--n_segments` | Number of video segments; equal to the number of local attention mechanisms. | 4 | None, int ≥ 2
`--pos_enc` | Type of the applied positional encoding. | 'absolute' | None, 'absolute', 'relative'
`--heads` | Number of heads of the global attention mechanism. | 8 | int > 0
`--n_epochs` | Number of training epochs. | 200 | int > 0
`--batch_size` | Size of the training batch, 20 for 'SumMe' and 40 for 'TVSum'. | 20 | 0 < int ≤ len(Dataset)
`--clip` | Gradient norm clipping parameter. | 5 | float 
`--lr` | Value of the adopted learning rate. | 5e-5 | float
`--l2_req` | Value of the regularization factor. | 1e-5 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 4
`--init_type` | Weight initialization method. | 'xavier' | None, 'xavier', 'normal', 'kaiming', 'orthogonal'
`--init_gain` | Scaling factor for the initialization methods. | None | None, float

## Model Selection and Evaluation 
<div align="justify">

The utilized model selection criterion relies on the post-processing of the calculated losses over the training epochs and enables the selection of a well-trained model by indicating the training epoch. To evaluate the trained models of the architecture and automatically select a well-trained model, define the [`dataset_path`](evaluation/compute_fscores.py#L25) in [`compute_fscores.py`](evaluation/compute_fscores.py) and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh). To run this file, specify:
 - [`base_path/exp$exp_num`](evaluation/evaluate_exp.sh#L6-L7): the path to the folder where the analysis results are stored,
 - [`$dataset`](evaluation/evaluate_exp.sh#L8): the dataset being used, and
 - [`$eval_method`](evaluation/evaluate_exp.sh#L9): the used approach for computing the overall F-Score after comparing the generated summary with all the available user summaries (i.e., 'max' for SumMe and 'avg' for TVSum).
```bash
sh evaluation/evaluate_exp.sh $exp_num $dataset $eval_method
```
For further details about the adopted structure of directories in our implementation, please check line [#6](evaluation/evaluate_exp.sh#L6) and line [#11](evaluation/evaluate_exp.sh#L11) of [`evaluate_exp.sh`](evaluation/evaluate_exp.sh). </div>

## Citations
If you find this code useful for your research, consider cite our paper:
```
@article{cardoso2025streamlined,
  title={Streamlined extended long short-term memory for video skimming},
  author={Cardoso, Leonardo Vilela and Soraggi, Barbara Hellen P and Guimar{\~a}es, Silvio Jamil F and Patroc{\'\i}nio Jr, Zenilton KG},
  journal={Pattern Recognition Letters},
  year={2025},
  publisher={Elsevier}
}

```
