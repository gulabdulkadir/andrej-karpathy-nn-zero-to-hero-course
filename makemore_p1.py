import os
import micrograd_from_scratch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt


words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
#27x27 tablo oluşturduk 0 la dolu
chars = sorted(list(set(''.join(words))))
# ''.join(...) ile bütün kelimeleri tek bir stringe koyar, daha sonra set(...) ile 
# tekrarlayan harfleri siler sadece unique olarak kalır,
# list(...) ile listeye çevirir her bir harf index alır artık sorted ile de sıralar 
# yeni hali ['a', 'b', ....'z']
stoi = {s:i+1 for i, s in enumerate(chars)}
# harflere 1 den başlatarak sıra numarası verdik 0'a da '.' koyduk
stoi['.'] = 0

itos = {i:s for s, i in stoi.items()}
# stoi nin tam tersidir


for w in words:
    chs =['.'] + list(w) + ['.']
    # her kelimenin başına ve sonuna . koy
    for ch1, ch2 in zip (chs, chs[1:]):
        # kaydırmalı eşleşme yapar böylece peşpeşe gelene her iki harf bir çift oluşturur
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

P = (N+1).float()
# ilerde log. alırken hata almamak için hepsine +1 yaptık
P /= P.sum(1, keepdims=True)
# normalizasyon yani p olasılığı vermek için her satırın toplamını bulduk, her bölmeyi de satır toplamına böldük 
# böylelikle her bir satırın olasılık toplamı 1 e eşit oldu


g = torch.Generator().manual_seed(83)
#hep aynı sonucu almak için seed li random generator

# for i in range(5):
#     out = []
#     ix = 0
#     #boş bir dizi oluşturduk ve biz stoi['.']'yı 0. indexe koymuştuk nokta ile başlamalı o yüzden nokta ile başlattık
#     while True:
        
#         p = P[ix]
#         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         #olasılığı yüksek olan harfin gelme olasılığının yüksek olduğu bir zart atışı yapar
#         out.append(itos[ix])
#         if ix == 0:
#             break
#     print(''.join(out))
        
log_likelihood = 0.0
n = 0

# for w in words:
# # eğer bir ismi loss func ile hesaplamak istiyorsak
# # for w in ["abdulkadir"]:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         #log(a) + log(b) = log(a*b)
#         n += 1
#         #log ile yaptık çünkü diyelim 0.1 * 0.0001 * 0.0000001 yaptığımızda sayı hassasiyetinden dolayı yok sayılacaktı
        
# print(f'{log_likelihood=}')
# nll = -log_likelihood
# # logaritma ile yaptığımızda ismin kalitesi arttıkça hesapladığımız sayı büyüyor (yani 0'a yaklaşıyor)
# # ama bizim loss fonksiyonumuzda kalite arttıkça loss azalmalı yani sayı küçülmeli bu yüzden de negatifini alıyoruz
# print(f'{nll=}')
# # ve normalizasyon
# print(f'{nll/n}')

# # biagramdan training set oluştur
# xs, ys = [], []
# for w in words[:1]:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         print(ch1, ch2)
#         xs.append(ix1)
#         ys.append(ix2)

# xs = torch.tensor(xs)
# ys = torch.tensor(ys)

# print(xs)
# print(ys)

# #xenc = [0, 5, 13, 13, 1]
import torch.nn.functional as F
# # sadeece istediğimiz sıradaki harfin ağırlığını almak için one hot encoding yapıyoruz
# xenc = F.one_hot(xs, num_classes=27).float() 
# print(xenc) 
# print(xenc.shape)
# # plt.imshow(xenc)
# #print(xenc.dtype) 

## karpathy (27, 1) olusturmmus ama sonucu 27, 27 lik gösteriyor ?!?!?
# W = torch.randn((27, 27), generator=g, requires_grad=True)
# # print(xenc @ W) # one coding matris ile ağırlık matrisini çarpıyoruz

# logits = xenc @ W   #log-counts

# # alttaki 2 satıra softmax denir
# counts = logits.exp()
# # exp fonksiyonu bütün negatifleri yok eder
# probs = counts / counts.sum(1, keepdims = True)
# # burda da olasılığa çeviriyoruz her satırın toplamı 1 oluyor

# ## satır toplam 1 mi kontrolü
# # satir_toplam = probs[0].sum()
# # print (satir_toplam)

# #loss fonksiyonu 
# loss = -probs[torch.arange(5), ys].log().mean()

# print ("loss:", loss.item())

# W.grad = None #gradları 0 a eşitle
# # backward ve update kısmı ağırlıklar için
# loss.backward()
# W.data += -50 * W.grad
# # -50 adım büyüklüğü alpha gibi

# #gradient descent 
# num = xs.nelement()

# 1. Tüm veriyi hazırla (words[:1] yerine words kullan)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))



 

        