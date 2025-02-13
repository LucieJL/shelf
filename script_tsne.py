import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

machine = '14-501-10'
n_compo_audio = 100
n_compo_embed = 40
perp = 30

print('Loading data...', end='')

if machine == '14-501-10':
    dur = 15
    audio = np.array(np.load(f'audio_{dur}_kerguelen2014.npy'))
    embed = np.array(torch.load(f'embed_{dur}_kerguelen2014.pt'))

elif machine == '14-501-11':
    dur = 299
    audio = np.array(torch.load(f'audio_{dur}_kerguelen2014.pt'))
    embed = np.array(torch.load(f'embed_{dur}_kerguelen2014.pt'))

else :
    dur = 640
    audio = np.array(torch.load(f'audio_{dur}_kerguelen2014.pt'))
    embed = np.array(torch.load(f'embed_{dur}_kerguelen2014.pt'))

print('done !')

if n_compo_audio is not None:
    audio_save_name = f'audio_tsne_pca{n_compo_audio}_dur{dur}_perp{perp}.pt'
    print('Computing PCA...', end='')
    pca = PCA(n_components=n_compo_audio)
    out_audio = pca.fit_transform(audio)
    print('done !')

if n_compo_embed is not None:
    embed_save_name = f'embed_tsne_pca{n_compo_audio}_dur{dur}_perp{perp}.pt'
    print('Computing PCA...', end='')
    pca = PCA(n_components=n_compo_embed)
    out_embed = pca.fit_transform(embed)
    print('done !')

else:
    out_audio = audio
    out_embed = embed
    audio_save_name = f'audio_tsne_out_dur{dur}_perp{perp}.pt'
    embed_save_name = f'embed_tsne_out_dur{dur}_perp{perp}.pt'

print('Computing TSNE...', end='')
tsne = TSNE(n_components=2, perplexity=perp)
final_audio = tsne.fit_transform(out_audio)
final_embed = tsne.fit_transform(out_embed)
print('done !')

print('Saving...', end='')
torch.save(torch.Tensor(final_audio), audio_save_name)
torch.save(torch.Tensor(final_embed), embed_save_name)
print('done !')