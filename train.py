import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration
from whole_process import process_directory
from utils import distance_loss

def train(args, model, train_loader, optimizer, scheduler, criterion,  model_save_path, pad_idx, device):
    model.to(device)
    model.train()
    print("Training Start")
    
    
    for epoch in range(args.epochs):
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target,vid_name = data
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)
            vid_name = vid_name

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type == 'i3d_transcript':
                inputs = (features, past_label,vid_name)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, past_label)


            _,_,vidnames = inputs
            outputs = model(inputs)
            
            losses = torch.tensor(0.0, device=device,requires_grad=True)
            lambda1, lambda2, lambda3, lambda4 = 1, 20, 20, 20
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, target_past_label, pad_idx)
                losses = losses + lambda1*loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                output = output.view(-1, C).to(device)
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total = cal_performance(output, target, pad_idx)
                acc = n_correct / n_total
                loss_class = loss.item()
                losses = losses + lambda2*loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)
                
                losses += lambda3*loss_dur
                epoch_loss_dur += loss_dur.item()

        # Inductive Bias
            current_action = outputs['seg']
            current_action = current_action[:, -1, :]
            normlize_current_action = F.softmax(current_action,dim=1)
            current_action = torch.argmax(normlize_current_action,dim=1)
            predicted_actions = torch.zeros(B, T, C - 1)
            one_hot_actions = F.one_hot(current_action, num_classes=C-1).float()
            current_action_probility = normlize_current_action[:,:C-1]
            predicted_actions[:,0,:] = one_hot_actions * current_action_probility
            
            for s in range(0, B):
                vidname = vidnames[s]
                base_dir = './datasets/50salads/ProbMatrix'
                npy_dir = os.path.join(base_dir, vidname.replace('.txt', '_transition_probabilities.npy'))
                transition_probabilities = np.load(npy_dir)

                for t in range(0, T):
                    if t == 0:
                        confitensor = predicted_actions[s, t, :]
                        confi = confitensor[confitensor != 0]

                        next_actions = []
             
                        action_probs = transition_probabilities[current_action[s]]
                        
                        next_action_probs = action_probs
                        max_idx = np.argmax(next_action_probs)
                        next_action_probs = np.zeros_like(next_action_probs)
                        next_action_probs[max_idx] = confi

                        if isinstance(next_action_probs, np.ndarray):
                            next_action_probs = torch.tensor(next_action_probs, dtype=torch.float32)

                       
                        predicted_actions[s, t, :] = next_action_probs

         
                        next_action = torch.argmax(next_action_probs).item()

                        next_actions.append(next_action)

                        current_action_predict = torch.tensor(next_actions, dtype=torch.int64)
                    else:
                        confitensor = predicted_actions[s, t-1, :]
                        confi = confitensor[confitensor!=0]

                        next_actions = []

                        action_probs = transition_probabilities[current_action_predict[0]]
                        next_action_probs = action_probs
                        max_idx = np.argmax(next_action_probs)
                        next_action_probs = np.zeros_like(next_action_probs)
                        next_action_probs[max_idx] = confi

                        if isinstance(next_action_probs, np.ndarray):
                            next_action_probs = torch.tensor(next_action_probs, dtype=torch.float32)

                        predicted_actions[s, t, :] = next_action_probs

                        next_action = torch.argmax(next_action_probs).item()

                        next_actions.append(next_action)
                        current_action_predict = torch.tensor(next_actions, dtype=torch.int64)

            # print(predicted_actions.shape)
            action_to_add = torch.full((B, T, C), 0,dtype=torch.float)
            action_to_add[:,:,:C-1] = predicted_actions

            action_to_add = action_to_add.to(device)

            output = outputs['action']
            output_action = F.softmax(output, dim=-1)

            # print(action_to_add.shape)
            # print(output_action.shape)
            d_loss = distance_loss(action_to_add,output_action)
            losses += lambda4*d_loss
            
            
            
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        scheduler.step()

        save_path = os.path.join(model_save_path)
        if epoch >= 30 :
            save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    return model
