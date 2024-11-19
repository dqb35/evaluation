import os
import time
import torch
import numpy as np
import argparse
import einops
from tqdm import tqdm
from torchvision.transforms import Resize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils_dqb.datasets_dqb import Dataset_weather
from utils_dqb.datasets_dqb import Dataset_SMD_pkl,Dataset_PSM_npy,Dataset_SWaT_npy
from utils_dqb.datasets_dqb import Dataset_MSL
from utils_dqb.metrics import metric
from utils_dqb.eval import evaluate
from vqvae_muse import VQGANModel, get_tokenizer_muse
from transformers import LlamaForCausalLM

interp_resize = Resize((256,256),interpolation=Image.BILINEAR,antialias=False)

dataset_dict ={"weather":Dataset_weather,"SMD":Dataset_SMD_pkl,"MSL":Dataset_MSL,"PSM":Dataset_PSM_npy,"SWaT":Dataset_SWaT_npy}


def resize_and_repeat(data):
    """
    input:
    data: tensor.size[seqs,h,w]
    output:
    tensor:[seqs,3,256,256]
    """
    # 转换size为256
    data_256_tensor = interp_resize(data)
    # 拓展RGB通道
    data_256_tensor_3 = einops.repeat(data_256_tensor.unsqueeze(1),'s 1 h w -> s c h w ',c=3)

    return data_256_tensor_3

def reshapedata_to_2d(data,channels,batch_size):
    """
    data:array(batch_size,channels1,channels2) channels1 = channels2
    return:
    data_2d:array(channels1,channels2*batch_size)
    """
    # 转置数据，使得形状变为 (variable_channels, time_steps, batch_size)
    data_transposed = data.transpose(1, 2, 0)  # 变为 (21, 21, 4)
    # 然后调整形状，将其变为 (variable_channels, time_steps * batch_size)
    data_2d = data_transposed.reshape(channels, channels * batch_size)
    return data_2d


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LVM_for_MTS_test')
    
    parser.add_argument('--data',type=str,default='SMD') # 数据集名称
    parser.add_argument('--channels',type=int,default=38) # 数据集的通道变量数 SMD:38 , PSM:25, SWaT:51
    parser.add_argument('--path',type=str,default='/mnt/data/Students/wangxt/Data/SMD/machine-1-1_test.pkl') # 数据路径
    parser.add_argument('--scale',type=bool,default=False)  # 数据读取时是否要数据缩放-minmaxscaler
    parser.add_argument('--inverse',type=bool,default=False) # 是否要反归一化

    parser.add_argument('--seqs',type=int,default=5) # 每个样本中的图片序列长度
    parser.add_argument('--preds',type=int,default=1) # 每个样本预测图片的数量

    parser.add_argument('--checkpoint',type=str,default='/mnt/data/Students/wangxt/LVM_ckpts') # LLaMA模型权重路径

    parser.add_argument('--batch_size',type=int,default=4) 


    args = parser.parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    out_resize = Resize((args.channels,args.channels),interpolation=Image.BILINEAR,antialias=False) #不同的数据集的c不同

    setting = ''
    # 加载数据集
    Data = dataset_dict[args.data]
    dataset = Data(path=args.path,seqs=args.seqs,preds=args.preds,scale=args.scale)
    print(f"{args.data}数据集数量:{len(dataset)}")
    data_test_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,drop_last=False)

    # 加载分词器
    tokenizer = get_tokenizer_muse()
    tokenizer.to(device)
    # 加载模型
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype= torch.float16).to(device) 

    preds = []
    trues = []
    preds_inverse = []
    trues_inverse = []

    # i = 0

    model.eval()
    with torch.no_grad():
        for batch_x,batch_y in tqdm(data_test_loader,desc='test_lvm_for_MTS'):
            start_time = time.time()
            
            batch_size,channels,_ = batch_x.shape
            seqs = args.seqs
            
            reshaped_data = batch_x.reshape(batch_size, channels, seqs, channels) # 先通过 reshape 调整形状，使得时间步长维度变成可以被 21 整除的多个切片
            final_data = reshaped_data.permute(0, 2, 1, 3) # 调整维度，得到最终形状 [4, 5, 21, 21]
            # 将 final_data 转换为 float32 类型
            final_data = final_data.to(torch.float32)

            # 插值扩展与通道扩展
            inputs = []
            for no_batch_data in final_data:
                data_rr = resize_and_repeat(no_batch_data) # torch.size[5,3,256,256]
                inputs.append(data_rr)
            inputs = torch.stack(inputs,dim=0).to(torch.float32).to(device) # torch.Size([4, 5, 3, 256, 256])

            # vqgan encode
            # tokenizer.encode输入的数据最多为4维[batch,channels,h,w]
            inputs_ids_list = []
            for no_batch_input in inputs:
                _,input_ids = tokenizer.encode(no_batch_input)
                input_ids = input_ids.view(1, -1) # torch.Size([1, 256*seqs])
                inputs_ids_list.append(input_ids)
            inputs_ids = torch.cat(inputs_ids_list,dim=0) # torch.Size([4, 1280]) 4即batch
            # print(f'inputs_ids.shape:{inputs_ids.shape}')

            new_frames = args.preds # 每个序列样本产生结果的个数
            top_p=1.0 # 用于 nucleus sampling,决定了在生成时选择的 token 概率的累积和.这个参数越小,生成的文本越保守
            temperature=1.0 # 控制生成的随机性.温度越高,生成的内容越随机;温度越低,生成的内容越确定性.
            """
            attention_mask:这个参数用于告诉模型哪些 tokens 是填充的.代码中使用了全 1 的张量,表示所有位置都参与注意力计算
            pad_token_id: 这个参数指定填充使用的 token ID
            do_sample: 如果为 True,模型会进行随机采样,而不是使用贪婪搜索
            suppress_tokens: 指定要抑制生成的 token IDs,可以用来避免生成特定的 token
            """
            outputs_ids = model.generate(
                input_ids=inputs_ids,
                attention_mask=torch.ones_like(inputs_ids),
                pad_token_id=8192,
                max_new_tokens=256 * new_frames,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                suppress_tokens=list(range(8192, model.vocab_size)),) # outputs_ids.shape:torch.Size([4, 1536])
            #  print(f'outputs_ids.shape:{outputs_ids.shape}') 

            # 截取结果
            new_tokens = []
            new_tokens.append(outputs_ids[:, -256 * new_frames:]) # [torch.Size([batch, 256])]
            new_tokens = torch.cat(new_tokens, dim=1).view(-1, 256) # torch.Size([batch, 256])
            # vqgan decode
            new_image = tokenizer.decode_code(new_tokens)
            # print(f'new_image.shape:{new_image.shape}') # torch.Size([batch, 3, 256, 256])
            y_grey = torch.mean(new_image,1,keepdim=True) # torch.Size([batch, 1, 256, 256])
            y = out_resize(y_grey).squeeze() # torch.Size([batch, 21, 21])
            # print(y.shape)
            y = y.detach().cpu().numpy() # (4,21,21)
            batch_y = batch_y.detach().cpu().numpy() # (4,21,21)
            if args.scale and args.inverse:
                # 3d to 2d
                batch_y_2d = reshapedata_to_2d(data=batch_y,channels=channels,batch_size=batch_size)
                y_2d = reshapedata_to_2d(y,channels=channels,batch_size=batch_size)
                # 反归一化
                batch_y_inverse = dataset.inverse_transform(batch_y_2d)
                y_inverse = dataset.inverse_transform(y_2d)
                # 记录结果
                preds_inverse.append(y_inverse)
                trues_inverse.append(batch_y_inverse)

            preds.append(y)
            trues.append(batch_y)

            end_time = time.time()
            onefor_time =end_time - start_time
            print(f"这次循环共跑了{args.batch_size}个样本,共用时{onefor_time}")


            # i += 1
            # if i ==2:
            #     break
            
    
    preds_array = np.concatenate(preds,axis=0)
    trues_array = np.concatenate(trues,axis=0)
    print(f'preds_array.shape:{preds_array.shape}')
    print(f'trues_array.shape:{trues_array.shape}') 

    if args.scale and args.inverse:
        preds_inverse_array = np.concatenate(preds_inverse,axis=1)
        trues_inverse_array = np.concatenate(trues_inverse,axis=1)
        print(f'preds_inverse_array.shape:{preds_inverse_array.shape}')
        print(f'trues_inverse_array.shape:{trues_inverse_array.shape}')



    mae,mse,rmse,mape,mspe = metric(preds_array,trues_array)
    print(f'mae:{mae},mse:{mse}')
    f = open("result_lvm_for_MTS.txt",'a')
    f.write(setting+"\n")
    f.write(f'mae:{mae},mse:{mse}')
    f.write('\n')
    f.write('\n')
    f.close()




    # 保存结果
    folder_path = f'./results/{args.data}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    np.save(folder_path + 'preds.npy',preds_array)
    np.save(folder_path + 'true.npy',trues_array)

    if args.scale and args.inverse:
        np.save(folder_path + 'preds_inverse.npy',preds_inverse_array)
        np.save(folder_path + 'true_inverse.npy',trues_inverse_array)




    
