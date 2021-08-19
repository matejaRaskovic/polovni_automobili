import os

images_folder = 'slike'
max_len = 0
max_pth = ''

for fldr in os.listdir(images_folder):
    pth = os.path.join(images_folder, fldr)
    if max_len < len(os.listdir(pth)):
        max_len = len(os.listdir(pth))
        max_pth = pth
    # max_len = max(max_len, len(os.listdir(pth)))

print(max_len)
print(max_pth)