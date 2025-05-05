from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode

def pad(data):  # PKCS7 padding
    pad_len = 16 - len(data) % 16
    return data + bytes([pad_len]) * pad_len

def unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

def encrypt(data, key):
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    enc = cipher.encrypt(pad(data))
    return b64encode(iv + enc).decode()

def decrypt(enc_data, key):
    raw = b64decode(enc_data)
    iv, enc = raw[:16], raw[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc))

# # Exemple
# if __name__ == "__main__":
#     key = get_random_bytes(16)
#     secret = b"data confidentielle"
#     enc = encrypt(secret, key)
#     print(enc)
#     print(decrypt(enc, key))