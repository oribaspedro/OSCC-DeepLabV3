def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    aux_loss=False
):
    for epoch in range(epochs):
        model.train()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)

            loss = criterion(outputs['out'], masks)

            if aux_loss:
                loss += 0.4 * criterion(outputs['aux'], masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")