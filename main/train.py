from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from dataset import train_dataset,val_dataset
from transformers import AutoModel, AutoProcessor, AutoConfig
from model import CheXagentSigLIPBinary
from utils import train_one_epoch, validate
import os

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
vision_full = AutoModel.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
).to(device, dtype)
vision_encoder = vision_full.vision_model
del vision_full


def train_model():

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)


    model = CheXagentSigLIPBinary(vision_encoder= vision_encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint_path= 'best_model.pth'
    
    # ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only= False)
    # model.load_state_dict(ckpt)
    model.to(device)
    

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.AdamW(
    #     [
    #         {"params": model.vision_encoder.parameters(), "lr": 3e-6},
    #         {"params": model.classifier.parameters(), "lr": 3e-4},
    #     ],
    #     weight_decay=1e-4
    # )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.vision_encoder.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4
    )
    # Training
    EPOCHS = 10
    start_epoch = 0
    best_val_acc = 0.0  # to store best accuracy
    best_val_loss = float("inf")

    # if os.path.exists("best_model_70_acc.pth"):
    #     checkpoint = torch.load("best_model_70_acc.pth", map_location=device)
    #     model.load_state_dict(checkpoint["model_state"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state"])
    #     start_epoch = checkpoint["epoch"] + 1
    #     best_val_loss = checkpoint["best_val_loss"]
    #     print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("-" * 50)

        # # ---- SAVE BEST MODEL ----
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), "best_model.pth")
        #     print(f"Best model updated with val_acc = {best_val_acc:.4f}")
        # # break

        ### changing saving the best model - 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "best_model.pth"
            )
            # torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model updated with loss = {best_val_loss:.4f}")
            print(f"validation_accuracy= {val_acc:.4f}")
        # break

    # ---- SAVE LAST MODEL ----
    torch.save(model.state_dict(), "last_model.pt")
    print("Last model saved as last_model.pt")


if __name__ == "__main__":
    train_model()