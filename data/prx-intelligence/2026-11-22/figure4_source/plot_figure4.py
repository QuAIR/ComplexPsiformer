"""Reproduce Figure 4: C6-averaged, orientation-aligned (a), original (b,c).

Requires NumPy, Matplotlib and PyMuPDF. Run in any directory. The existing lower
panels (shared colorbar and panel c) are preserved as vector PDF artwork.
"""
from pathlib import Path
import hashlib
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
import numpy as np
import pymupdf

HERE=Path(__file__).resolve().parent
META=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
with np.load(HERE/'figure4_data.npz') as z:
    ma,mb=z['nu_one_third_c6'],z['nu_three_tenths']
    lattice_a,lattice_b=z['lattice1'],z['lattice2']
    angle=float(z['display_rotation_degrees'])


def draw_map(ax,g,lattice,rotation_degrees):
    n=len(g)
    theta=np.deg2rad(np.arange(30,390,60))
    poly=np.stack((np.cos(theta),np.sin(theta)),axis=-1)*np.linalg.norm(lattice[0])/np.sqrt(3)
    phi=np.deg2rad(rotation_degrees)
    R=np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    # The hexagon is a fixed display crop. Tile the rotated periodic field far
    # enough to cover this crop; rotate the data mesh but not the outline.
    uv=poly@np.linalg.inv(lattice@R)
    lower=np.floor((uv.min(0)+.5)*n).astype(int)-1
    upper=np.ceil((uv.max(0)+.5)*n).astype(int)+1
    ii,jj=[np.arange(lo,hi) for lo,hi in zip(lower,upper)]
    u,v=[np.arange(lo,hi+1)/n-.5 for lo,hi in zip(lower,upper)]
    points=np.stack(np.meshgrid(u,v,indexing='ij'),axis=-1)@lattice@R
    im=ax.pcolormesh(points[...,0],points[...,1],g[np.ix_(ii%n,jj%n)],
        cmap=cmap,norm=Normalize(0,2.5),shading='flat',linewidth=0,
        antialiased=False,edgecolors='none',rasterized=True)
    patch=Polygon(poly,closed=True,facecolor='none',edgecolor='#83929A',linewidth=.47)
    ax.add_patch(patch); im.set_clip_path(patch)
    ax.set(aspect='equal',xticks=[-6,0,6],yticks=[-6,0,6],
           xlim=(-1.01*radius,1.01*radius),ylim=(-1.018*radius,1.018*radius))
    ax.set_xlabel(r'$x/\ell_B$',labelpad=.8)


plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
    'axes.labelsize':7.1,'axes.titlesize':8,'xtick.labelsize':6.1,'ytick.labelsize':6.1,
    'mathtext.fontset':'dejavusans','text.color':'#24323B','axes.labelcolor':'#273540',
    'axes.edgecolor':'#77858B','xtick.color':'#526069','ytick.color':'#526069','axes.linewidth':.55,
    'xtick.major.size':2.2,'ytick.major.size':2.2,'xtick.major.width':.55,'ytick.major.width':.55,
    'xtick.major.pad':1.4,'ytick.major.pad':1.4,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
cmap=LinearSegmentedColormap.from_list('pairs',[(0,'#357780'),(.18,'#A0C9C9'),
    ((11/12)/2.5,'#FAF7ED'),(.55,'#F5D19C'),(.76,'#E29751'),(1,'#9F4824')],N=512)
fig=plt.figure(figsize=(META['figure_width_pt']/72,META['figure_height_pt']/72))
radius=np.linalg.norm(lattice_b[0])/np.sqrt(3)
for rect,title,g,L,deg,letter in zip(([.114,.596,.383,.32],[.581,.596,.383,.32]),
    (r'$\nu=1/3$',r'$\nu=3/10$'),(ma,mb),(lattice_a,lattice_b),(angle,0),'ab'):
    ax=fig.add_axes(rect)
    draw_map(ax,g,L,deg)
    ax.set_title(title,pad=6)
    if letter=='a':ax.set_ylabel(r'$y/\ell_B$',labelpad=.4)
    else:ax.tick_params(labelleft=False)
    ax.text(-.14,1.075,letter,transform=ax.transAxes,fontweight='bold',fontsize=8.7,va='bottom')
top_buffer=io.BytesIO()
fig.savefig(top_buffer,format='pdf',dpi=600)
plt.close(fig)
top=pymupdf.open(stream=top_buffer.getvalue(),filetype='pdf')
lower=pymupdf.open(HERE/'figure4_lower_panels.pdf')
out=pymupdf.open()
page=out.new_page(width=META['figure_width_pt'],height=META['figure_height_pt'])
page.show_pdf_page(page.rect,top,0)
page.show_pdf_page(pymupdf.Rect(0,META['lower_panels_y_start_pt'],page.rect.width,page.rect.height),lower,0)
output=HERE.parent/'figs/figure4_continuum.pdf'
out.save(output,garbage=4,deflate=True)
page.get_pixmap(matrix=pymupdf.Matrix(3,3),alpha=False).save(HERE/'figure4_preview.png')
print(json.dumps({'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                  'rotation_degrees':angle,'mean_a':float(ma.mean()),'mean_b':float(mb.mean())}))
