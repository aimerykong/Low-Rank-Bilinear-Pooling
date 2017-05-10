function inter=getInter(iseg, segLen, up)
    inter=(iseg-1)*segLen+1 : min(iseg*segLen, up);
end