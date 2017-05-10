function out=getDimOfBlob(resp)
    sz=size(resp);
    out=sz(1:(end-1));
    out=num2cell(out);
    if 0
        fprintf('Warning: special case for batchsize==1 \n');
        out=num2cell(sz);
    end
end