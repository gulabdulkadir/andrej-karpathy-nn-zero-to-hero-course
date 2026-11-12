import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


#kelimelerin olduğu dosyayı okuma modunda açtık
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
# ''.join(words) bütün kelimeleri tek bir satırmış gibi birleştirir
# set(...) metindeki tekrarlayan kelimeleri siler
# list listeye çevirir
# sorted(...) listeyi sıralar alfabetik sıraya göre

# string-to-integers
stoi = {s:i+1 for i, s in enumerate(chars)}
# enumerate önce indexi sonra harfi verir
stoi['.'] = 0
# 0. indexe . koyduk

# integers-to-string
itos = {s: i for i, s in stoi.items()}

# dataset i inşa edelim
block_size = 3
def build_dataset(words):

    # kaç karakterden sonra bir sonrakini tahmin edicez
    X, Y = [], []
    # data ve target
    for w in words:
        context  = [0] * block_size
        # her ismin sonuna '.' koyduk
        for ch in w + '.':
            ix = stoi[ch]
            # ix: listedeki isimlerin harflerinin integera dönüşmüş hali
            X.append(context)
            # X e ... ekledik her ismin başına ki 3 harften sonra tahmin yapacağımız için
            Y.append(ix)
            # artık Y ise listedeki isimlerin integera dönüşmüş hali
            context = context[1:] + [ix]
            # context[1:] en baştaki eleman silinmiş halde isim ix harflerin integer haliydi yani böyle olcak context = ..1 gibi?
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    # torch nesnesine dönüştürmek: hantallığı yok eder, bellekte yan yana tutar (bloklar halinde), 
    # python nesneleri sadece cpu ve ram üzerinde yaşayabilirken torch olunca gpu da çalıştırabiliriz
    # requires_grad = True ise sonuca olan etkisini otomatik hesaplar (chain kuralı ile), zengin matematik kütüphanesi ve bellek verimliğliği sağlar
    return X, Y

random.seed(83)
random.shuffle(words)
# harfleri karıştırdık ki aynı harfle başlayan isimler peşpeşe gelip modeli yanıltmasın
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
# n1 e kadar yani %80 tranining set 

Xdev, Ydev = build_dataset(words[n1:n2])
# n1 den n2 arası yani %90 - %80 = %10 Development yani geliştirme bölümümüz, hiperparametreleri burda belirleyeceğiz

Xte, Yte = build_dataset(words[n2:])
# n2 den sonraki kısım yine %10 test kısmı
print(Xtr.shape, Ytr.shape,"\n", Xdev.shape, Ydev.shape, "\n", Xte.shape, Yte.shape)

g = torch.Generator().manual_seed(83)

C = torch.randn((27, 10), generator=g)
# embedding tablosu artık her harf bir sayı değil, 10 boyutlu kordinat sistemindeki bir nokta

W1 = torch.randn((30, 200), generator=g)
# block_size = 3 yaptık yani her 3 harften sonra gelen harfi tahmin edicez 10 boyutlu olduğu içinde 10 x 3 = 30
# 200 ise çıkan sayıdır bu bir hiperparametredir

b1 = torch.randn(200, generator=g)
# bias

W2 = torch.rand((200, 27), generator=g)
# bu çıkış katmanını tekrar 27 ye düşürecek ağırlık matrisi 

b2 = torch.rand(27, generator=g)
# 2. bias

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
# -3 ten 0 a eşit aralıkta 1000 tane sayı üretir
# learning rate exponets
lrs = 10**lre
# learning rates:
# başlangıç değeri 10^-3 yani 0.001, bitiş değeri 10^0 yani 1 bu aralıkta 1000 tane sayı üret diyor
# bunu uygun parametreyi bulmak için yapacağız

lri = []
lossi = []
stepi = []

for i in range(200000):
    ix = torch.randint(0, Xtr.shape[0], (32, ))
    # 0 başlangıç idexi, xtr.shape[0] xtranining setinin boyutunda ilkini yani satır sayısını aldı, 32 kaç tane rastgele sayı seçileceği
    # biz bütün veriyi almak yerine (trainnig sette 182bin veri vardı) 32şerli parça parça alıyoruz ki hesaplama maaliyeti azalsın
    
    # forward pass
    emb = C[Xtr[ix]] # (32, 3, 10)
    # embedding sonucunda bizim elimizde 3 lü liste var 32 x 3 x 10
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
    # biz 3 x 10 u birleştirmesini istiyoruz ve 32 x 30 luk bir liste veriyor bize
    # bunnu da 30 x 200 lük ağırlıkla çarpıp bias ekliyoruz ve bunu tanh fonk. sokuyoruz, tanh doğrusallığa eğim katar
    logits = h @ W2 + b2 # (32, 27)
    # 30 x 200 lük listeyi tekrar 200 x 27 ile çarpup alfabe boyutuna düşürdük
    loss = F.cross_entropy(logits, Ytr[ix])
    # 27 tane  ham skoru alır ve toplamları 1 olacak şekilde olasılığa çevirir, doğru harfin olasılığı ne kadar düşükse o kadar yüksek
    # ceza puanı keser
    # backward pass
    for p in parameters:
        # her seferinde gradları 0 lamamız gerekiyor
        p.grad = None
    # loss fonk. için backward
    loss.backward()
    
    # update
    lr = 0.1 if i < 100000 else 0.01
    
    for p in parameters:
        p.data += -lr * p.grad
    # parametre güncellemeleri
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())


plt.plot(stepi, lossi)

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
block_size = 3
for _ in range(20):
  
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      # emb.view(1, -1) de ilk boyutum kesinlikle 1 olsun diyoruz, 2. parametreyi -1  yaparak geri kalanını sen sığdır diyoruz
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      # olasılığa çevirdik
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # olasılıklara göre rastgele seçim yapar
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))










        