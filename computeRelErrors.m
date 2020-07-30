function e = computeRelErrors(X, ranks)

e = zeros(length(ranks),1);
for ri = 1:length(ranks)
    fprintf('checking rank=%d...\n',ranks(ri));
    Xhat = cp_construct(SECSI(X, ranks(ri), 'BM'));
    e(ri) = norm(Xhat(:) - X(:)) ./ norm(X(:));
end

end