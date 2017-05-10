function net=addDropout(net, rate)
    net.layers{end+1}=struct('type','dropout','rate', rate);
end