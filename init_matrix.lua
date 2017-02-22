function rand_ortho(N,M,range)
 --print(N,M,range)
 A = torch.Tensor(N,M):rand(N,M)*2*range - range;
 U,s,V=torch.svd(A);
 return V*torch.eye(M,M)*U:t();


end
