%% Huffman-Based Distribution Matcher for Probabilistic Constellation Shaping
% Implements a variable-rate Huffman codec that maps binary data to shaped
% symbol sequences with optimal entropy coding.
% 
% Advantages over constant-composition:
%  - Higher spectral efficiency (variable rate)
%  - Reduced latency (smaller block buffers)
%  - Simpler for online/streaming applications
%
% Trade-offs:
%  - Exact probability matching only in steady-state
%  - Requires synchronization on codeword boundaries

clc; clear; close all;

%% Configuration
pTarget = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05];  % Target symbol probabilities
alphabet = 1:length(pTarget);  % Symbol indices
M = length(alphabet);

% Normalize probabilities
pTarget = pTarget / sum(pTarget);

fprintf('=== Huffman-Based Distribution Matcher ===\n\n');
fprintf('Target probabilities: '); fprintf('%.4f ', pTarget); fprintf('\n\n');

%% Build Huffman tree and codebook
[huffmanTree, codeBook] = build_huffman_tree(pTarget);
fprintf('Huffman Codebook:\n');
for m = 1:M
    fprintf('  Symbol %d (p=%.4f): %s (len=%d bits)\n', ...
        m, pTarget(m), codeBook{m}, length(codeBook{m}));
end

% Calculate entropy and code efficiency
entropy = -sum(pTarget .* log2(pTarget + eps));
avgCodeLen = sum(pTarget .* cellfun(@length, codeBook));
efficiency = entropy / avgCodeLen;

fprintf('\nEntropy H(X):        %.4f bits/symbol\n', entropy);
fprintf('Avg code length:     %.4f bits/symbol\n', avgCodeLen);
fprintf('Code efficiency:     %.4f (ideal: 1.0)\n\n', efficiency);

%% Encode random symbol sequence
numSymbols = 20;
symbolSeq = randsample(alphabet, numSymbols, true, pTarget);

fprintf('Input symbol sequence: ');
fprintf('%d ', symbolSeq); fprintf('\n');

% Encode symbols to bitstream
bitStream = [];
for s = symbolSeq
    bitStream = [bitStream, codeBook{s}];
end

fprintf('Encoded bitstream (%d bits): %s\n', length(bitStream), ...
    sprintf('%d', bitStream));
fprintf('Actual rate: %.4f bits/symbol\n', length(bitStream) / numSymbols);

%% Decode bitstream back to symbols
[decodedSymbols, remainingBits] = huffman_decode(bitStream, huffmanTree, numSymbols);

fprintf('\nDecoded symbol sequence: ');
fprintf('%d ', decodedSymbols); fprintf('\n');
fprintf('Remaining bits: %s\n', sprintf('%d', remainingBits));
fprintf('Decoding match: %s\n\n', isequal(symbolSeq, decodedSymbols) && 'YES' || 'NO');

%% Encode binary message to shaped symbols
fprintf('=== Binary Message Encoding ===\n\n');
numBits = 30;
msgBits = randi([0, 1], 1, numBits);
fprintf('Input message (%d bits): %s\n', numBits, sprintf('%d', msgBits));

% Convert bits to symbols using Huffman decoder
shapedSymbols = bits_to_huffman_symbols(msgBits, huffmanTree);
fprintf('Shaped symbol sequence: '); fprintf('%d ', shapedSymbols); fprintf('\n');
fprintf('Num symbols: %d\n', length(shapedSymbols));

% Reconstruct bits from symbols
reconstructedBits = huffman_symbols_to_bits(shapedSymbols, codeBook);
fprintf('Reconstructed bits: %s\n', sprintf('%d', reconstructedBits));
fprintf('Encoding success: %s\n\n', isequal(msgBits, reconstructedBits(1:numBits)) && 'YES' || 'NO');

%% Statistical analysis
fprintf('=== Statistical Distribution Analysis ===\n\n');
nTrials = 1000;
symbolCounts = zeros(1, M);

for trial = 1:nTrials
    nSymbols = randsample(5:20, 1);  % Random block size
    syms = randsample(alphabet, nSymbols, true, pTarget);
    for s = syms
        symbolCounts(s) = symbolCounts(s) + 1;
    end
end

observedProbs = symbolCounts / sum(symbolCounts);

fprintf('Target vs. Observed Distribution:\n');
fprintf('Symbol | Target   | Observed | Δ\n');
fprintf('-------|----------|----------|--------\n');
for m = 1:M
    fprintf('  %d    | %.4f   | %.4f   | %.4f\n', m, pTarget(m), observedProbs(m), ...
        abs(pTarget(m) - observedProbs(m)));
end

%% Visualization
figure('Position', [100, 100, 1400, 600]);

% Subplot 1: Huffman tree visualization
subplot(2, 3, 1);
visualize_huffman_tree(huffmanTree);
title('Huffman Tree Structure');
axis off;

% Subplot 2: Target vs observed probabilities
subplot(2, 3, 2);
x = 1:M;
bar(x - 0.2, pTarget, 0.4, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
bar(x + 0.2, observedProbs, 0.4, 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
xlabel('Symbol'); ylabel('Probability');
legend('Target', 'Observed');
title('Distribution Matching');
grid on; ylim([0, max(pTarget)*1.3]);

% Subplot 3: Code length distribution
subplot(2, 3, 3);
codeLengths = cellfun(@length, codeBook);
bar(alphabet, codeLengths, 'FaceColor', [0.2 0.8 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
xlabel('Symbol'); ylabel('Huffman Code Length (bits)');
title('Variable-Rate Code Lengths');
grid on;

% Subplot 4: Cumulative bitstream
subplot(2, 3, 4);
cumulativeBits = cumsum(bitStream);
plot(cumulativeBits, 'LineWidth', 1.5, 'Color', [0.2 0.4 0.8]);
grid on; xlabel('Bit Position'); ylabel('Cumulative Bits');
title('Encoded Bitstream Accumulation');

% Subplot 5: Rate variation over multiple blocks
subplot(2, 3, 5);
rates = [];
for trial = 1:30
    nSyms = randi([10, 30]);
    syms = randsample(alphabet, nSyms, true, pTarget);
    nBits = 0;
    for s = syms
        nBits = nBits + length(codeBook{s});
    end
    rates = [rates, nBits / nSyms];
end
plot(rates, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
yline(entropy, '--', 'Entropy', 'LineWidth', 2, 'Color', [0.8 0.2 0.2]);
grid on; xlabel('Block'); ylabel('Actual Rate (bits/symbol)');
title('Variable Rate per Block');
ylim([entropy - 0.5, max(rates) + 0.2]);

% Subplot 6: Performance metrics
subplot(2, 3, 6);
axis off;
textStr = {
    sprintf('Entropy H(X): %.4f bits/symbol', entropy);
    sprintf('Avg Code Length: %.4f bits/symbol', avgCodeLen);
    sprintf('Code Efficiency: %.4f', efficiency);
    sprintf('Max Code Length: %d bits', max(codeLengths));
    sprintf('Min Code Length: %d bits', min(codeLengths));
    '';
    sprintf('Avg Rate (trials): %.4f bits/symbol', mean(rates));
    sprintf('Rate Std Dev: %.4f', std(rates));
};
text(0.1, 0.9, textStr, 'VerticalAlignment', 'top', 'FontFamily', 'monospace', ...
    'FontSize', 10);

sgtitle('Huffman-Based Distribution Matcher for PCS');

%% ========== Helper Functions ==========

function [tree, codeBook] = build_huffman_tree(probs)
    % Build Huffman tree from probability vector
    % tree.symbol: symbol index (leaf) or [] (internal node)
    % tree.left, tree.right: child subtrees
    % tree.prob: node probability
    
    M = length(probs);
    
    % Initialize leaf nodes
    nodes = cell(M, 1);
    for m = 1:M
        nodes{m} = struct('symbol', m, 'prob', probs(m), 'left', [], 'right', []);
    end
    
    % Build tree bottom-up
    while length(nodes) > 1
        % Sort by probability
        [~, idx] = sort(cellfun(@(x) x.prob, nodes));
        nodes = nodes(idx);
        
        % Merge two smallest nodes
        node1 = nodes{1};
        node2 = nodes{2};
        parentProb = node1.prob + node2.prob;
        parent = struct('symbol', [], 'prob', parentProb, 'left', node1, 'right', node2);
        
        nodes = [nodes(3:end), {parent}];
    end
    
    tree = nodes{1};
    
    % Generate codebook via tree traversal
    codeBook = cell(M, 1);
    traverse_tree(tree, '', codeBook);
end

function traverse_tree(node, code, codeBook)
    % Recursively traverse Huffman tree to build codebook
    if ~isempty(node.symbol)
        % Leaf node
        codeBook{node.symbol} = code;
    else
        % Internal node
        if ~isempty(node.left)
            traverse_tree(node.left, [code, '0'], codeBook);
        end
        if ~isempty(node.right)
            traverse_tree(node.right, [code, '1'], codeBook);
        end
    end
end

function [symbols, remaining] = huffman_decode(bitStream, tree, numSymbols)
    % Decode bitstream to symbols using Huffman tree
    symbols = [];
    remaining = bitStream;
    
    for k = 1:numSymbols
        node = tree;
        while isempty(node.symbol)
            if isempty(remaining)
                error('Incomplete bitstream for Huffman decoding');
            end
            if remaining(1) == '0'
                node = node.left;
            else
                node = node.right;
            end
            remaining = remaining(2:end);
        end
        symbols = [symbols, node.symbol];
    end
    
    if ischar(remaining)
        remaining = remaining - '0';
    end
end

function symbols = bits_to_huffman_symbols(bits, tree)
    % Convert binary stream to symbol sequence by traversing Huffman tree
    bitStr = sprintf('%d', bits);
    symbols = [];
    idx = 1;
    
    while idx <= length(bitStr)
        node = tree;
        while isempty(node.symbol) && idx <= length(bitStr)
            if bitStr(idx) == '0'
                node = node.left;
            else
                node = node.right;
            end
            idx = idx + 1;
        end
        if ~isempty(node.symbol)
            symbols = [symbols, node.symbol];
        end
    end
end

function bits = huffman_symbols_to_bits(symbols, codeBook)
    % Convert symbol sequence back to bits using codebook
    bitStr = '';
    for s = symbols
        bitStr = [bitStr, codeBook{s}];
    end
    bits = bitStr - '0';
end

function visualize_huffman_tree(tree)
    % Simple ASCII visualization of Huffman tree structure
    % (More sophisticated plot would require custom graphics)
    
    function depth = tree_depth(node)
        if isempty(node.symbol) && (~isempty(node.left) || ~isempty(node.right))
            left_d = 0;
            right_d = 0;
            if ~isempty(node.left)
                left_d = tree_depth(node.left);
            end
            if ~isempty(node.right)
                right_d = tree_depth(node.right);
            end
            depth = max(left_d, right_d) + 1;
        else
            depth = 1;
        end
    end
    
    d = tree_depth(tree);
    text(0.5, 0.9, sprintf('Huffman Tree Depth: %d levels', d), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.5, 0.7, 'Left branch: 0, Right branch: 1', ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Style', 'italic');
end
