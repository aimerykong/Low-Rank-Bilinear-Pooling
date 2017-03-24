% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
    for j=1:numel(net.layers(i).params)
        par = net.layers(i).params{j} ;
        format(end+1,1:3) = {'single', size(par), sprintf('l%d_%d',i,j)} ;
    end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, ...
    'Format', format, ...
    'Repeat', numGpus, ...
    'Writable', true) ;

