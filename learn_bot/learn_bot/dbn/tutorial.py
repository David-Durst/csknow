import pyAgrum as gum
import pyAgrum.lib.image as gumimage

bn=gum.BayesNet('WaterSprinkler')
print(bn)

c=bn.add(gum.LabelizedVariable('c','cloudy ?',2))
print(c)

s, r, w = [ bn.add(name, 2) for name in "srw" ]
print (s,r,w)
print (bn)

bn.addArc(c,s)

for link in [(c,r),(s,w),(r,w)]:
    bn.addArc(*link)
print(bn)
gumimage.export(bn, "models/test_export.png")
x = bn.cpt(c).fillWith([0.4,0.6])

from pyAgrum.lib.notebook import *
showP