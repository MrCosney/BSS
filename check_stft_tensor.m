clear; clc; set(0,'DefaultFigureWindowStyle','docked');
%% Load simulated dataset
fs = 16000;
addpath('amsbss/convolutive_datasets');
%% RT60 = 0.9s
fname = 'stationary_ss_rt60-0.9_PYROOM.mat';
loadedData = load(fname);
% Load mixed audio
x_mixed = loadedData.mixed_ss;
% Load original audio (not mixed)
x_original = loadedData.original_rir_ss;
%% Normalize
x_mixed = x_mixed./(max(abs(x_mixed(:)))); 
%% Compute STFT tensor
[X, window] = STFT(x_mixed,1024,512,'hamming');
%%
X1 = X(:, :, 1);
r = randn(size(X1)) + 1i*randn(size(X1));
%% Plot singular values (frequency x time)
sv_X1 = svd(X1); sv_X1 = sv_X1 ./ max(sv_X1);
sv_X1abs = svd(abs(X1).^2); sv_X1abs = sv_X1abs ./ max(sv_X1abs);
sv_r = svd(r); sv_r = sv_r ./ max(sv_r);
sv_X1ang = svd(angle(X1)); sv_X1ang = sv_X1ang ./ max(sv_X1ang);
figure('Name','Singular values (frequency x time)');
subplot(2,1,1);
plot(sv_r); hold on; plot(sv_X1); plot(sv_X1abs); plot(sv_X1ang);
subplot(2,1,2);
semilogy(sv_r); hold on; semilogy(sv_X1); semilogy(sv_X1abs); semilogy(sv_X1ang);
%% Try CPD
datasets = {
    'stationary_ss_rt60-0.05_TIMIT_dist-1.0m.mat',...
    'stationary_ss_rt60-0.05_TIMIT_dist-2.5m.mat',...
    'stationary_ss_rt60-0.1_TIMIT_dist-1.0m.mat',...
    'stationary_ss_rt60-0.1_TIMIT_dist-2.5m.mat',...
    'stationary_ss_rt60-0.2_TIMIT_dist-1.0m.mat',...
    'stationary_ss_rt60-0.2_TIMIT_dist-2.5m.mat',...
    'stationary_ss_rt60-0.9_PYROOM.mat'};
%% Raw STFT tensor
figure('Name','CP relative error vs rank (raw)');
axLin = subplot(2,1,1); xlabel('Rank'); ylabel('Rel error, %'); 
ylim([0 100]); hold on;
axLog = subplot(2,1,2); xlabel('Rank'); ylabel('Rel error, %');
ylim([1 100]); axLog.YScale = 'log'; hold on;
for dsi = 1:length(datasets)
    dataSetName = datasets{dsi};
    fprintf('Data set: %s\n',dataSetName);
    %% Load data set
    loadedData = load(dataSetName);
    % Load mixed audio
    x_mixed = loadedData.mixed_ss;
    %% Compute STFT
    x_mixed = x_mixed./(max(abs(x_mixed(:)))); 
    %% Compute STFT tensor
    [X, ~] = STFT(x_mixed,1024,512,'hamming');
    sz = sort(size(X));
    ranks = 1:sz(2);
    %% Compute CP errors
    re = computeRelErrors(X, ranks);
    %% Plot
    plot(axLin, ranks, 100*re, 'DisplayName',dataSetName);
    plot(axLog, ranks, 100*re, 'DisplayName',dataSetName);
end
legend(axLin); legend(axLog);
%% Squared STFT tensor
figure('Name','CP relative error vs rank (squared)');
axLin = subplot(2,1,1); xlabel('Rank'); ylabel('Rel error, %'); 
ylim([0 100]); hold on;
axLog = subplot(2,1,2); xlabel('Rank'); ylabel('Rel error, %');
ylim([1 100]); axLog.YScale = 'log'; hold on;
for dsi = 1:length(datasets)
    dataSetName = datasets{dsi};
    fprintf('Data set: %s\n',dataSetName);
    %% Load data set
    loadedData = load(dataSetName);
    % Load mixed audio
    x_mixed = loadedData.mixed_ss;
    %% Compute STFT
    x_mixed = x_mixed./(max(abs(x_mixed(:)))); 
    %% Compute STFT tensor
    [X, ~] = STFT(x_mixed,1024,512,'hamming');
    X = abs(X).^2;    
    sz = sort(size(X));
    ranks = 1:sz(2);
    %% Compute CP errors
    re = computeRelErrors(X, ranks);
    %% Plot
    plot(axLin, ranks, 100*re, 'DisplayName',dataSetName);
    plot(axLog, ranks, 100*re, 'DisplayName',dataSetName);
end
legend(axLin); legend(axLog);
%% W0 of ILRMA
figure('Name','CP relative error vs rank (W0-ILRMA)');
axLin = subplot(2,1,1); xlabel('Rank'); ylabel('Rel error, %'); 
ylim([0 100]); hold on;
axLog = subplot(2,1,2); xlabel('Rank'); ylabel('Rel error, %');
ylim([1 100]); axLog.YScale = 'log'; hold on;
for dsi = 1:length(datasets)
    dataSetName = datasets{dsi};
    fprintf('Data set: %s\n',dataSetName);
    %% Load data set
    loadedData = load(dataSetName);
    % Load mixed audio
    x_mixed = loadedData.mixed_ss;
    %% Compute STFT
    x_mixed = x_mixed./(max(abs(x_mixed(:)))); 
    %% Compute W0    
    [~, W0] = ilrma_bss(x_mixed, 50, 1024, 10);
    %% Inverse W0
    A0 = zeros(size(W0));
    for i=1:size(W0,3)
        A0(:,:,i) = inv(W0(:,:,i));
    end
    %%
    sz = sort(size(A0));
    ranks = 1:6;
    %% Compute CP errors
    re = computeRelErrors(A0, ranks);
    %% Plot
    plot(axLin, ranks, 100*re, 'DisplayName',dataSetName);
    plot(axLog, ranks, 100*re, 'DisplayName',dataSetName);
end
legend(axLin); legend(axLog);



