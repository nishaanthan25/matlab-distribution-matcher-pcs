%% 16-QAM Probabilistic Constellation Shaping with AWGN Channel
% Full simulation of shaped constellation transmission over AWGN channel
% including distribution matching, modulation, noise, and receiver detection.
%
% Features:
%  - 16-QAM Gray-coded constellation
%  - Constant-composition distribution matcher (CCDM)
%  - AWGN channel with configurable SNR
%  - MMSE symbol detection
%  - BER and SER performance curves
%  - Constellation visualization at different SNR levels

clc; clear; close all;

%% System Parameters
fprintf('=== 16-QAM Shaped PCS with AWGN Channel Simulation ===\n\n');

% Constellation and shaping parameters
M = 16;                              % 16-QAM
constellation = generate_gray_16qam();
pTarget = [0.20, 0.15, 0.25, 0.10, ...
           0.12, 0.08, 0.05, 0.03, ...
           0.02, 0.01, 0.01, 0.01, ...
           0.01, 0.01, 0.00, 0.00];  % Shaped distribution (non-uniform)
pTarget = pTarget / sum(pTarget);     % Normalize

% Verify probability length matches constellation
if length(pTarget) ~= M
    error('Probability vector length must equal constellation size');
end

% Block parameters
N = 64;                              % Block length (symbols)
SNR_dB = -5:1:15;                    % SNR sweep range (dB)
nTrials = 100;                       % Monte Carlo trials per SNR

% CCDM parameters
counts = round(pTarget * N);
counts = fix_to_sum(counts, N);

fprintf('Constellation: %d-QAM (Gray-coded)\n', M);
fprintf('Block length N: %d symbols\n', N);
fprintf('Number of trials: %d\n', nTrials);
fprintf('SNR range: [%.1f, %.1f] dB\n\n', min(SNR_dB), max(SNR_dB));

% Display target distribution
fprintf('Target Symbol Distribution:\n');
fprintf('  Index | Probability\n');
fprintf('  ------|------------\n');
for m = 1:M
    fprintf('   %2d  |  %.4f\n', m-1, pTarget(m));
end
fprintf('\n');

%% Build CCDM Lookup Table
fprintf('Building CCDM lookup table...\n');
seqTable = build_ccdm_table_fast(counts);
numSequences = size(seqTable, 1);
numBits = floor(log2(numSequences));
capacity = numBits / N;

fprintf('  Number of valid sequences: %d\n', numSequences);
fprintf('  Encodable bits per block: %d\n', numBits);
fprintf('  Spectral efficiency: %.4f bits/symbol\n\n', capacity);

%% Performance Metrics Initialization
ber = zeros(size(SNR_dB));
ser = zeros(size(SNR_dB));
snr_linear = 10.^(SNR_dB / 10);

%% Main Simulation Loop
fprintf('Running AWGN simulation...\n');

for snr_idx = 1:length(SNR_dB)
    snr_db = SNR_dB(snr_idx);
    snr_lin = snr_linear(snr_idx);
    
    % Noise standard deviation
    noise_std = sqrt(1 / (2 * snr_lin));  % Assuming unit average power
    
    bit_errors = 0;
    symbol_errors = 0;
    total_bits = 0;
    total_symbols = 0;
    
    for trial = 1:nTrials
        % Generate random information bits
        msgBits = randi([0, 1], 1, numBits);
        
        % Encode bits to shaped symbol sequence via CCDM
        symbolIndices = dm_encode_16qam(msgBits, seqTable);
        txSymbols = constellation(symbolIndices);
        
        % AWGN Channel
        noise = noise_std * (randn(size(txSymbols)) + 1j * randn(size(txSymbols)));
        rxSymbols = txSymbols + noise;
        
        % MMSE Symbol Detection (nearest constellation point)
        detectedIndices = zeros(size(rxSymbols));
        for k = 1:length(rxSymbols)
            [~, detectedIndices(k)] = min(abs(rxSymbols(k) - constellation));
        end
        
        % Decode detected symbols back to bits
        decodedBits = dm_decode_16qam(detectedIndices, seqTable, counts);
        
        % Count bit errors
        bit_errors = bit_errors + sum(msgBits ~= decodedBits);
        total_bits = total_bits + numBits;
        
        % Count symbol errors
        symbol_errors = symbol_errors + sum(symbolIndices ~= detectedIndices);
        total_symbols = total_symbols + N;
    end
    
    ber(snr_idx) = bit_errors / total_bits;
    ser(snr_idx) = symbol_errors / total_symbols;
    
    fprintf('  SNR = %+6.1f dB | BER = %.2e | SER = %.2e\n', snr_db, ber(snr_idx), ser(snr_idx));
end

fprintf('\nSimulation complete.\n\n');

%% Theoretical Performance (Uniform 16-QAM for reference)
ber_uniform = zeros(size(SNR_dB));
for idx = 1:length(SNR_dB)
    snr_lin = 10.^(SNR_dB(idx) / 10);
    % Approximate BER for 16-QAM: Q(sqrt(4.8 * Eb/N0))
    Eb_N0 = snr_lin * log2(M);
    ber_uniform(idx) = 0.5 * erfc(sqrt(0.2 * Eb_N0));
end

%% Visualization
figure('Position', [100, 100, 1600, 900]);

% Subplot 1: BER Comparison
subplot(2, 3, 1);
semilogy(SNR_dB, ber, 'bo-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Shaped 16-QAM (PCS)');
hold on;
semilogy(SNR_dB, ber_uniform, 'r^--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Uniform 16-QAM');
grid on; xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)');
title('BER Performance vs SNR');
legend('Location', 'northeast'); ylim([1e-5, 1]);

% Subplot 2: SER Comparison
subplot(2, 3, 2);
semilogy(SNR_dB, ser, 'go-', 'LineWidth', 2, 'MarkerSize', 6);
grid on; xlabel('SNR (dB)'); ylabel('Symbol Error Rate (SER)');
title('SER Performance vs SNR');

% Subplot 3: Constellation at different SNR levels
SNR_plot = [0, 5, 10];
colors = {'red', 'orange', 'green'};

subplot(2, 3, 3);
hold on;
for snr_val = SNR_plot
    snr_lin = 10^(snr_val / 10);
    noise_std = sqrt(1 / (2 * snr_lin));
    
    % Generate one noisy constellation
    symbolIndices = 1:M;
    txSyms = constellation(symbolIndices);
    noise = noise_std * (randn(1, M) + 1j * randn(1, M));
    rxSyms = txSyms + noise;
    
    idx = find(SNR_plot == snr_val);
    scatter(real(rxSyms), imag(rxSyms), 50, colors{idx}, 'filled', 'Alpha', 0.5, ...
        'DisplayName', sprintf('SNR=%d dB', snr_val));
end

scatter(real(constellation), imag(constellation), 200, 'k', 'x', 'LineWidth', 2, ...
    'DisplayName', 'Ideal');
grid on; axis equal; xlabel('I'); ylabel('Q');
title('Received Constellation at Different SNR');
legend('Location', 'best', 'FontSize', 9);

% Subplot 4: Target vs Observed Distribution
subplot(2, 3, 4);
observed = histcounts(symbolIndices, 0.5:1:(M+0.5)) / N;
bar(0:M-1, pTarget, 'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
hold on;
bar(0:M-1, observed, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
xlabel('Constellation Index'); ylabel('Probability');
title('Target vs Observed Distribution');
legend('Target', 'Observed'); grid on; ylim([0, max(pTarget)*1.3]);

% Subplot 5: Noise Variance Impact
subplot(2, 3, 5);
noise_vars = 1 ./ (2 * snr_linear);
plot(SNR_dB, noise_vars, 'LineWidth', 2.5, 'Color', [0.2 0.4 0.8]);
grid on; xlabel('SNR (dB)'); ylabel('Noise Variance');
title('Noise Power vs SNR');
set(gca, 'YScale', 'log');

% Subplot 6: Performance Metrics Table
subplot(2, 3, 6);
axis off;

textStr = sprintf(['=== PCS 16-QAM AWGN Simulation ===\n' ...
    'Block Length: %d symbols\n' ...
    'Info Rate: %.4f bits/symbol\n' ...
    'Constellation: Gray 16-QAM\n' ...
    'Trials: %d per SNR\n\n' ...
    'Performance at SNR = 10 dB:\n' ...
    '  BER: %.2e\n' ...
    '  SER: %.2e\n\n' ...
    'Gain vs Uniform:\n' ...
    '  Approx: +1-2 dB at BER~1e-3'], ...
    N, capacity, nTrials, ber(find(SNR_dB==10)), ser(find(SNR_dB==10)));

text(0.05, 0.95, textStr, 'VerticalAlignment', 'top', 'FontFamily', 'monospace', ...
    'FontSize', 10, 'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'black', 'Margin', 5);

sgtitle(sprintf('16-QAM Probabilistic Constellation Shaping | Spectral Efficiency: %.4f bits/symbol', capacity), ...
    'FontSize', 12, 'FontWeight', 'bold');

%% Additional Analysis: SNR Gain at Target BER
fprintf('SNR Gain Analysis:\n');
fprintf('-----------------\n');

target_ber = 1e-3;
idx_pcs = find(ber <= target_ber, 1);
idx_uniform = find(ber_uniform <= target_ber, 1);

if ~isempty(idx_pcs) && ~isempty(idx_uniform)
    snr_pcs = SNR_dB(idx_pcs);
    snr_uniform = SNR_dB(idx_uniform);
    gain = snr_uniform - snr_pcs;
    fprintf('At BER = %.0e:\n', target_ber);
    fprintf('  Shaped PCS:  %.1f dB\n', snr_pcs);
    fprintf('  Uniform 16-QAM: %.1f dB\n', snr_uniform);
    fprintf('  Gain: %.2f dB\n\n', gain);
else
    fprintf('Target BER not achieved in simulation range.\n\n');
end

%% ========== Helper Functions ==========

function constellation = generate_gray_16qam()
    % Generate 16-QAM with Gray coding
    % Returns vector of 16 complex constellation points
    
    % 4x4 grid normalized to unit average power
    levels = [-3, -1, 1, 3];
    scaling = sqrt(10 / 9);  % Normalize to Es = 1
    
    constellation = [];
    gray_indices = [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10];
    
    for idx = gray_indices
        I = levels(bitand(idx, 3) + 1);
        Q = levels(bitshift(bitand(idx, 12), -2) + 1);
        constellation = [constellation; (I + 1j*Q) * scaling];
    end
end

function counts = fix_to_sum(counts, N)
    % Adjust count vector to sum exactly to N
    while sum(counts) < N
        [~, idx] = max(counts);
        counts(idx) = counts(idx) + 1;
    end
    while sum(counts) > N
        [~, idx] = min(counts);
        if counts(idx) > 0
            counts(idx) = counts(idx) - 1;
        end
    end
end

function seqTable = build_ccdm_table_fast(counts)
    % Build constant-composition distribution matcher lookup table
    % seqTable: numSequences x N matrix, each row is valid symbol sequence
    
    M = length(counts);
    N = sum(counts);
    seqTable = [];
    
    function recurse(prefix, remaining)
        if length(prefix) == N
            seqTable = [seqTable; prefix];
            return;
        end
        
        for s = 1:M
            if remaining(s) > 0
                nextRemaining = remaining;
                nextRemaining(s) = nextRemaining(s) - 1;
                recurse([prefix, s], nextRemaining);
            end
        end
    end
    
    recurse([], counts);
end

function symbolIndices = dm_encode_16qam(bits, seqTable)
    % Encode bit sequence to shaped symbol indices via CCDM
    
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
    
    if msgIndex > numMsgs
        error('Invalid message index');
    end
    
    symbolIndices = seqTable(msgIndex, :);
end

function bits = dm_decode_16qam(symbolIndices, seqTable, counts)
    % Decode shaped symbol indices back to bit sequence
    
    numMsgs = size(seqTable, 1);
    numBits = floor(log2(numMsgs));
    
    % Search for matching sequence in table
    for row = 1:numMsgs
        if isequal(seqTable(row, :), symbolIndices)
            msgIndex = row - 1;  % Convert to 0-based
            bits = dec2bin(msgIndex, numBits) - '0';
            return;
        end
    end
    
    % If no exact match found, return zeros (error case)
    warning('No matching sequence found in CCDM table');
    bits = zeros(1, numBits);
end
