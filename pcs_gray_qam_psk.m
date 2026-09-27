%% Probabilistic Constellation Shaping with Gray-Coded QAM/PSK
% Implements a constant-composition distribution matcher using Gray-coded
% QAM and PSK constellations with optimal bit-to-symbol mapping.

clc; clear; close all;

%% Configuration
% Target symbol probabilities (shaping filter output)
pTarget = [0.15, 0.20, 0.30, 0.20, 0.10, 0.05];
N = 100;  % Block length

% Constellation: 64-QAM Gray-coded
constellation = generate_gray_qam(64);
M = length(constellation);

% Verify probability vector length matches constellation
if length(pTarget) ~= M
    error('pTarget length must match constellation size');
end

% Normalize probabilities
pTarget = pTarget / sum(pTarget);

% Build target count vector for the block
counts = fix_to_sum_pcs(round(pTarget * N), N);

%% Generate lookup table for constant-composition DM
fprintf('Building constant-composition distribution matcher...\n');
seqTable = build_ccdm_table_fast(counts);
numMsgs = size(seqTable, 1);
numBits = floor(log2(numMsgs));

fprintf('Block length N = %d\n', N);
fprintf('Constellation size M = %d\n', M);
fprintf('Capacity = %.4f bits/symbol\n', numBits / N);
fprintf('Number of valid sequences = %d\n', numMsgs);
fprintf('Encodable bits per block = %d\n\n', numBits);

%% Test encoding/decoding
msgBits = randi([0, 1], 1, numBits);
fprintf('Input message bits: %s (decimal: %d)\n', ...
    sprintf('%d', msgBits), bin2dec(sprintf('%d', msgBits)));

% Encode
shapedIndices = dm_encode_qam(msgBits, seqTable);
shapedSymbols = constellation(shapedIndices);

% Decode
decodedBits = dm_decode_qam(shapedSymbols, constellation, seqTable, counts);
decodedMsg = bin2dec(sprintf('%d', decodedBits));

fprintf('Decoded message bits: %s (decimal: %d)\n', ...
    sprintf('%d', decodedBits), decodedMsg);
fprintf('Match: %s\n\n', isequal(msgBits, decodedBits) && 'YES' || 'NO');

%% Verify distribution matching
obsIndices = shapedIndices;
obsProbs = histcounts(obsIndices, 0.5:1:(M+0.5)) / N;

fprintf('Target probabilities:\n');
fprintf('  '); fprintf('%.4f ', pTarget); fprintf('\n');
fprintf('Observed probabilities:\n');
fprintf('  '); fprintf('%.4f ', obsProbs); fprintf('\n');
fprintf('Error (L∞): %.4f\n\n', max(abs(pTarget - obsProbs)));

%% Visualization
figure('Position', [100, 100, 1400, 500]);

% Subplot 1: Constellation with probabilities
subplot(1, 3, 1);
scatter(real(constellation), imag(constellation), 200, pTarget, 'filled');
colormap(gca, parula);
grid on; axis equal; xlabel('I'); ylabel('Q');
title('Gray-coded 64-QAM with Target Distribution');
colorbar; caxis([0, max(pTarget)]);

% Subplot 2: Symbol indices over block
subplot(1, 3, 2);
plot(shapedIndices, 'o-', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on; xlabel('Symbol position'); ylabel('Constellation index');
title('Shaped Symbol Block');

% Subplot 3: Probability histogram
subplot(1, 3, 3);
x = 1:M;
bar(x, pTarget, 'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
hold on;
bar(x, obsProbs, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
xlabel('Constellation index'); ylabel('Probability');
legend('Target', 'Observed');
title('Distribution Matching');
grid on; ylim([0, max(pTarget)*1.2]);

sgtitle(sprintf('PCS with Gray-coded 64-QAM | Capacity: %.4f bits/symbol', numBits/N));

%% ========== Helper Functions ==========

function constellation = generate_gray_qam(M)
    % Generate M-ary Gray-coded QAM constellation
    % For M = 4, 16, 64, 256, etc.
    
    k = log2(M);
    if mod(k, 2) ~= 0 || k < 2
        error('M must be a power of 4 (4, 16, 64, ...)');
    end
    
    m = sqrt(M);  % m x m grid
    
    % Create Gray code indices for I and Q
    grayI = graycode_sequence(log2(m));
    grayQ = graycode_sequence(log2(m));
    
    % Normalize to [-sqrt(M/2), sqrt(M/2)]
    scaling = sqrt(3 * M / (2 * (M - 1)));
    constellation = [];
    
    for i = 1:m
        for q = 1:m
            I = (2 * grayI(i) - m + 1) * scaling;
            Q = (2 * grayQ(q) - m + 1) * scaling;
            constellation = [constellation; I + 1j*Q];
        end
    end
end

function gray_seq = graycode_sequence(n)
    % Generate Gray code sequence for n bits (0 to 2^n - 1)
    gray_indices = bitxor(0:2^n-1, floor((0:2^n-1)/2));
    gray_seq = gray_indices + 1;  % 1-indexed
end

function counts = fix_to_sum_pcs(counts, N)
    % Adjust count vector to sum exactly to N
    counts = counts(:);
    while sum(counts) < N
        [~, idx] = max(counts);
        counts(idx) = counts(idx) + 1;
    end
    while sum(counts) > N
        [~, idx] = min(counts);
        counts(idx) = max(counts(idx) - 1, 0);
    end
end

function seqTable = build_ccdm_table_fast(counts)
    % Build CCDM lookup table more efficiently using recursion with memoization
    M = numel(counts);
    N = sum(counts);
    seqTable = zeros(0, N);
    
    function recurse(prefix, remaining, depth)
        if depth == N
            seqTable(end+1, :) = prefix;
            return;
        end
        
        for s = 1:M
            if remaining(s) > 0
                nextRemaining = remaining;
                nextRemaining(s) = nextRemaining(s) - 1;
                recurse([prefix, s], nextRemaining, depth + 1);
            end
        end
    end
    
    recurse([], counts, 0);
end

function shapedIndices = dm_encode_qam(bits, seqTable)
    % Encode bit sequence to shaped symbol indices
    numMsgs = size(seqTable, 1);
    numBits = floor(log2(numMsgs));
    
    if numel(bits) > numBits
        error('Input bits exceed encodable capacity');
    end
    
    % Zero-pad if needed
    if numel(bits) < numBits
        bits = [bits, zeros(1, numBits - numel(bits))];
    end
    
    % Convert binary to decimal index
    msgIndex = bin2dec(sprintf('%d', bits)) + 1;
    shapedIndices = seqTable(msgIndex, :);
end

function bits = dm_decode_qam(symbols, constellation, seqTable, counts)
    % Decode shaped symbols to bit sequence
    numMsgs = size(seqTable, 1);
    numBits = floor(log2(numMsgs));
    
    % Quantize symbols to nearest constellation point
    symbolIndices = zeros(1, numel(symbols));
    for i = 1:numel(symbols)
        [~, idx] = min(abs(symbols(i) - constellation));
        symbolIndices(i) = idx;
    end
    
    % Find matching sequence in table
    for row = 1:numMsgs
        if isequal(seqTable(row, :), symbolIndices)
            msgIndex = row - 1;
            bits = dec2bin(msgIndex, numBits) - '0';
            return;
        end
    end
    
    error('No matching sequence found');
end
