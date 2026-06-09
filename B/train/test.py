outputs = model(images)

loss = criterion(outputs, masks)

optimizer.zero_grad()

loss.backward()

optimizer.step()