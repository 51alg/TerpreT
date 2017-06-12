function [x,fval] = runLP(lpFile)

    % loads A, b, objective, lb and ub
    load(lpFile)

    Aeq = makeSparse(toDouble1Index(A));
    beq = makeSparse(toDouble1Index(b));
    objective = makeSparse(toDouble1Index(objective));
    lb = makeSparse(toDouble1Index(lb));
    ub_minus_1 = makeSparse(toDouble1Index(ub_minus_1));

    spy(Aeq);
    
    ub = 1 - ub_minus_1;
    
    % we are minimising
    obj = -objective;

    % solve
    options = optimoptions(@linprog,'Algorithm','interior-point');
    disp(' started solver...')
    tic;
    [x, fval] = linprog(obj,[],[],Aeq,beq,lb,ub,[],options);
    toc;

end

function s = toDouble1Index(s)
    s.values = double(s.values);
    s.dims = double(s.dims);
    if (numel(s.values)>0)
        s.values(:,1:(end-1)) = s.values(:,1:(end-1)) + 1; % 1 indexing
    end
end

function sp = makeSparse(s)
    if numel(s.values)>0
        switch numel(s.dims)
            case 2
                sp = sparse(s.values(:,1), s.values(:,2), s.values(:,3), s.dims(1), s.dims(2)); 
            case 1
                sp = sparse(ones(size(s.values,1),1),s.values(:,1),s.values(:,2),1,s.dims(1));
            otherwise
                error('Do not know how to make sparse matrix with %i dims',numel(s.dims))
        end
    else
	    switch numel(s.dims)
            case 2
                sp = sparse(s.dims(1), s.dims(2)); 
            case 1
                sp = sparse(1,s.dims(1));
            otherwise
                error('Do not know how to make sparse matrix with %i dims',numel(s.dims))
        end
    end
end