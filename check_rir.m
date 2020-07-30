function check_rir(rir)

rir = permute(rir, [3 1 2]);
rir_fd = fft(rir, max(size(rir)));
ranks = 1:7;
e = computeRelErrors(rir, ranks);
e_fd = computeRelErrors(rir_fd, ranks);
%%
figure('Name','RIR rank check');
ax1 = subplot(2,1,1); hold on; xlim([0 100]);
xlabel('rank'); ylabel('Relative error, %');
ax2 = subplot(2,1,2); ax2.YScale = 'log'; hold on;
xlabel('rank'); ylabel('Relative error, %'); xlim([0.01 100]);

plot(ax1, ranks, 100*e);
plot(ax2, ranks, 100*e);
plot(ax1, ranks, 100*e_fd);
plot(ax2, ranks, 100*e_fd);

end


