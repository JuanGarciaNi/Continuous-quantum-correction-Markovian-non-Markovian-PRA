function ToddNewPlots_Fast6()
% 6-qubit Liouville simulation (dim 4096)
% System qubits: (a1 a2 a3) = qubits 1..3
% Ancilla/bath qubits: (alpha1 alpha2 alpha3) = qubits 4..6
%\rho=| a1 a2 a3 alpha1 alpha2 alpha3 b1 b2 b3 beta1 beta2 beta3
% Generator: M = alpha*L + kappa*D + eta*(Phi⊗I_anc - I)

    % ---------------- USER SETTINGS ----------------
    alpha = 1;

    %kappa = 1;
    %etas = [0 1 2 4 10];

    eta = 1;
    kappas = [0 1 2 4 10];

    t_values = 0:0.1:10;     % smooth curves

    RelTol = 1e-7;
    AbsTol = 1e-10;
    % ------------------------------------------------

    N = 4096;
    % Initial vector v (your 8 basis indices)
    v = zeros(N,1);
    v([1,66,131,261,196,326,391,456]) = 1;

    % Plot RGB colors
    cols = [ ...
        0.00 0.45 0.74;
        0.85 0.33 0.10;
        0.93 0.69 0.13;
        0.49 0.18 0.56;
        0.47 0.67 0.19];

    figure; hold on;

%{
%====================== Eta Plots ============================ %
    for e = 1:numel(etas)
        eta = etas(e);
        fprintf('\n====================================================\n');
        fprintf('eta = %g\n', eta);
        fprintf('Building M (4096x4096)...\n');
        M = MatrSuperOp6_sparse(alpha, kappa, eta);
        fprintf('Done building M. nnz(M) = %d\n', nnz(M));

        % IMPORTANT: do NOT "shift spectrum" (kills steady states).
        try
            lam = eigs(M, 1, 'largestreal');
            fprintf('max real eig(M) ~ %.3e\n', real(lam));
        catch
            fprintf('max real eig(M): eigs failed\n');
        end

        opts = odeset('RelTol', RelTol, 'AbsTol', AbsTol, 'Jacobian', M);
        y = v;
        Fcurve = zeros(size(t_values));

        for k = 1:numel(t_values)
            if k > 1
                t0 = t_values(k-1);
                t1 = t_values(k);
                [~, Y] = ode15s(@(t,yy) M*yy, [t0 t1], y, opts);
                y = Y(end,:).';
            end
           
        Fk = real((v' * y)/8);

        % ---------- CLAMP FOR PLOTTING ----------
        % keeps the plot sane if small numerical overshoot happens
        Fk = max(0, min(1, Fk));
        % ---------------------------------------
            Fcurve(k) = Fk;
            fprintf('t = %6.3f   Fidelity = %.12f\n', t_values(k), Fk);
        end
        plot(t_values, Fcurve, 'LineWidth', 2.0, 'Color', cols(e,:));
    end

    % Plot styling
    xlim([min(t_values) max(t_values)]);
    ylim([0 1]);
    xlabel('\alpha t','FontSize',18);
    ylabel('Fidelity','FontSize',18);
    box on;
    set(gca,'FontSize',18,'LineWidth',1.5);

    L = legend(arrayfun(@(x)sprintf('\\eta/\\alpha=%g',x),etas,'UniformOutput',false), ...
               'Location','northeast');
    L.Box = 'on';
    L.LineWidth = 1.0;
    L.FontSize = 18;
    set(gcf,'Color','w');
end
%}
    

%====================== Kappa Plots ============================ %
    for e = 1:numel(kappas)
        kappa = kappas(e);
        fprintf('\n====================================================\n');
        fprintf('eta = %g\n', eta);
        fprintf('Building M (4096x4096)...\n');
        M = MatrSuperOp6_sparse(alpha, kappa, eta);
        fprintf('Done building M. nnz(M) = %d\n', nnz(M));

        % IMPORTANT: do NOT "shift spectrum" (kills steady states).
        try
            lam = eigs(M, 1, 'largestreal');
            fprintf('max real eig(M) ~ %.3e\n', real(lam));
        catch
            fprintf('max real eig(M): eigs failed\n');
        end

        opts = odeset('RelTol', RelTol, 'AbsTol', AbsTol, 'Jacobian', M);
        y = v;
        Fcurve = zeros(size(t_values));

        for k = 1:numel(t_values)
            if k > 1
                t0 = t_values(k-1);
                t1 = t_values(k);
                [~, Y] = ode15s(@(t,yy) M*yy, [t0 t1], y, opts);
                y = Y(end,:).';
            end
           
        Fk = real((v' * y)/8);

        % ---------- CLAMP FOR PLOTTING ----------
        % keeps the plot sane if small numerical overshoot happens
        Fk = max(0, min(1, Fk));
        % ---------------------------------------
            Fcurve(k) = Fk;
            fprintf('t = %6.3f   Fidelity = %.12f\n', t_values(k), Fk);
        end
        plot(t_values, Fcurve, 'LineWidth', 2.0, 'Color', cols(e,:));
    end

    % Plot styling
    xlim([min(t_values) max(t_values)]);
    ylim([0 1]);
    xlabel('\alpha t','FontSize',18);
    ylabel('Fidelity','FontSize',18);
    box on;
    set(gca,'FontSize',18,'LineWidth',1.5);

    L = legend(arrayfun(@(x)sprintf('\\kappa/\\alpha=%g',x),kappas,'UniformOutput',false), ...
               'Location','northeast');
    L.Box = 'on';
    L.LineWidth = 1.0;
    L.FontSize = 18;
    set(gcf,'Color','w');
end


function x0 = int2bits12(i)
    x0 = logical(bitget(uint16(i), 12:-1:1));
end

function idx = bits2idx12(b)
    idx = 1 + double(b) * (2.^(11:-1:0)).';
end


function M = MatrSuperOp6_sparse(alpha, kappa, eta)
    N = 4096;
    M = spalloc(N, N, 350000);

    for i = 0:N-1
        x0 = int2bits12(i);  % 1x12 logical bits (MSB first)

        col = alpha * liouv6_fast(x0) + ...
              kappa * DisSigmam6_fast(x0) + ...
              eta   * GammaCorr6_fast(x0);

        M(:, i+1) = col;

        if mod(i,512)==0 && i>0
            fprintf('  built column %d / %d\n', i, N);
        end
    end

    M = M.'; % your original transpose convention
end

function v = liouv6_fast(x0)
    N = 4096;

    m1 = logical([1 0 0 1 0 0 0 0 0 0 0 0]);
    m2 = logical([0 0 0 0 0 0 1 0 0 1 0 0]);
    m3 = logical([0 1 0 0 1 0 0 0 0 0 0 0]);
    m4 = logical([0 0 0 0 0 0 0 1 0 0 1 0]);
    m5 = logical([0 0 1 0 0 1 0 0 0 0 0 0]);
    m6 = logical([0 0 0 0 0 0 0 0 1 0 0 1]);

    idx = zeros(6,1);
    val = zeros(6,1);

    idx(1)=bits2idx12(xor(x0,m1)); val(1)=-1i;
    idx(2)=bits2idx12(xor(x0,m2)); val(2)=+1i;
    idx(3)=bits2idx12(xor(x0,m3)); val(3)=-1i;
    idx(4)=bits2idx12(xor(x0,m4)); val(4)=+1i;
    idx(5)=bits2idx12(xor(x0,m5)); val(5)=-1i;
    idx(6)=bits2idx12(xor(x0,m6)); val(6)=+1i;

    v = sparse(idx, ones(6,1), val, N, 1);
end

function v = DisSigmam6_fast(x0)
    N = 4096;
    idx = [];
    val = [];

    function add(bits, dv)
        idx(end+1,1) = bits2idx12(bits);
        val(end+1,1) = dv;
    end

    % ancilla-1 lowering: alpha1/beta1 = x0(4)/x0(10)
    if x0(4) && x0(10)
        b = x0; b(4)=false; b(10)=false;
        add(b, +1);
    end
    if x0(4),  add(x0, -1/2); end
    if x0(10), add(x0, -1/2); end

    % ancilla-2 lowering: alpha2/beta2 = x0(5)/x0(11)
    if x0(5) && x0(11)
        b = x0; b(5)=false; b(11)=false;
        add(b, +1);
    end
    if x0(5),  add(x0, -1/2); end
    if x0(11), add(x0, -1/2); end

    % ancilla-3 lowering: alpha3/beta3 = x0(6)/x0(12)
    if x0(6) && x0(12)
        b = x0; b(6)=false; b(12)=false;
        add(b, +1);
    end
    if x0(6),  add(x0, -1/2); end
    if x0(12), add(x0, -1/2); end

    if isempty(idx)
        v = sparse(N,1);
    else
        v = sparse(idx, ones(numel(idx),1), val, N, 1);
    end
end

function v = GammaCorr6_fast(x0)
    N = 4096;
    idx = [];
    val = [];

    function add(bits, dv)
        idx(end+1,1) = bits2idx12(bits);
        val(end+1,1) = dv;
    end

    a = x0(1:3);        % system ket bits
    b = x0(7:9);        % system bra bits

    % Determine Kraus-pair label for a and b (1..4) and their outputs (000 or 111)
    [la, ga] = phi_pair_and_output(a);
    [lb, gb] = phi_pair_and_output(b);

    if la == lb
        outBits = [ga, x0(4:6), gb, x0(10:12)];  % ancillas unchanged
        add(outBits, +1);
    else
        % so Gamma = (Phi - I) will be just -|a><b| here.
    end

    add(x0, -1);

    v = sparse(idx, ones(numel(idx),1), val, N, 1);
end

function [label, g] = phi_pair_and_output(bits3)
    % Pairs: (000,111), (100,011), (010,101), (001,110)
    % Output is 000 for the first element of the pair, 111 for its complement element.
    b = double(bits3(:).'); % 1x3 numeric
    n = b*[4;2;1];
    switch n
        case 0  % 000
            label = 1; g = logical([0 0 0]);
        case 7  % 111
            label = 1; g = logical([1 1 1]);

        case 4  % 100
            label = 2; g = logical([0 0 0]);
        case 3  % 011
            label = 2; g = logical([1 1 1]);

        case 2  % 010
            label = 3; g = logical([0 0 0]);
        case 5  % 101
            label = 3; g = logical([1 1 1]);

        case 1  % 001
            label = 4; g = logical([0 0 0]);
        case 6  % 110
            label = 4; g = logical([1 1 1]);

        otherwise
            error('Impossible 3-bit value');
    end
end
