import torch.nn.functional as F
import torch 

def attention_loss(pred_attn, gt_attn):
    """
    pred_attn: [B, 1024] (already sums to 1)
    gt_attn:   [B, 1024] (normalized)
    """
    pred_attn = pred_attn + 1e-6
    gt_attn = gt_attn + 1e-6

    return F.kl_div(
        pred_attn.log(),
        gt_attn,
        reduction="batchmean"
    )

def train_one_epoch(model, train_loader, optimizer, criterion, device,lambda_attn=0.1):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    count = 0 

    for image, label, attn_gt, has_bbox in train_loader:
        image = image.to(device)
        # print(image.shape)
        # view = view.to(device)
        # sex = sex.to(device)
        label = label.float().unsqueeze(1).to(device)  # [B,1]
        attn_gt = attn_gt.to(device)
        has_bbox = has_bbox.to(device)
        # print(label.shape)

        optimizer.zero_grad()

        outputs,pool_attn  = model(image)              # [B,1] logits
        cls_loss = criterion(outputs, label)
        if has_bbox.any():
            pred_attn = pool_attn[has_bbox]
            gt_attn = attn_gt[has_bbox]

            attn_loss = attention_loss(pred_attn, gt_attn)
        else:
            attn_loss = torch.tensor(0.0, device=device)
        loss = cls_loss + lambda_attn * attn_loss
        # print(outputs.shape,outputs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == label).sum().item()
        # print(f"label.size(0){label.size(0)}")
        total_samples += label.size(0)

        if count% 1000 ==0:
            print("Step", count)
        count+=1

        # break

    return total_loss / total_samples, total_correct / total_samples



def validate(model, val_loader, criterion, DEVICE):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    # count = 0

    with torch.no_grad():

        for image, label, attn_gt, has_bbox in val_loader:
            image = image.to(DEVICE)
            # view = view.to(DEVICE)
            # sex = sex.to(DEVICE)
            label = label.float().unsqueeze(1).to(DEVICE)
            
            outputs,pool_attn  = model(image)     
            loss = criterion(outputs, label)

            total_loss += loss.item()* label.size(0)

            # Accuracy
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)
            # print("val batch", count)
            # count+=1

            # break

    return total_loss / total_samples, total_correct / total_samples
