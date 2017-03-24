function net=addReLU(net, name)
    net.layers{end+1}=struct('type', 'relu', 'name', name);
end