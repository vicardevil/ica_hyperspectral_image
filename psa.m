clc
clear all;
for ii=1:20
a=read_ENVIimagefile('yyc200');
a=a/4095;
save('shuju.mat','a');
b=mean(a, 1);
[m1,n1,p1]=size(a);
t1=m1*n1;
data11=reshape(a,t1,p1);
[pc,score,latent,tsquare]=pca(data11);
feature_after_PCA=score(:,1:8);
img=reshape(feature_after_PCA,m1,n1,8);
a=img;
[m,n,p]=size(a);
t=m*n;
data11=reshape(a,t,p);
data1=data11';
mean_data=mean(data1,2);
data_mean=data1-mean_data;
co=1/64*(data1*data1');
[V,D] = eig(co);
F=V*sqrt(D)*V';
B_hat=F'*data_mean;
for i=1:p+1
    P(:,:,i)=eye(p);
end
U=[];
dimA=size(B_hat);
dimB=size(B_hat);
C=zeros([dimA(1)*dimB(1),dimA(2)]);
tic
for i=1:dimA(2)
    C(:,i)=kron(B_hat(:,i),B_hat(:,i));
end
ten1=B_hat*C'/40000;
for i=1:p
    S1(:,:,i)=ten1(:,(i-1)*p+1:i*p);
end
for i=1:p
    k=1;
    u(:,i,k)=rand(p,1);
    u(:,i,k)=u(:,i,k)/norm(u(:,i,k));  
    m=1;
   while m>=1e-3&&k<=1000
       K=P(:,:,i)*u(:,i,k);
       u(:,i,k+1)=tensor_mul(tensor_mul(tensor_mul(S1,K,1),P(:,:,i),2),K,3);
       u(:,i,k+1)=u(:,i,k+1)/norm(u(:,i,k+1));
       m=norm(u(:,i,k+1)-u(:,i,k));
       k=k+1;
   end
U(:,i)=u(:,i,k);
P(:,:,i+1)=eye(p)-U*((U'*U)^(-1))*U';
end
tt1(ii)=toc
Y=U'*B_hat;
Y=Y';
IMG=reshape(Y,200,200,8);
% for i=1:8
%     subplot(2,4,i);
%     imshow(mapminmax(IMG(:,:,i)));
% end
for i=1:8
    subplot(2,4,i);
    imshow((IMG(:,:,i)-min(min(IMG(:,:,i)))/max(max(IMG(:,:,i)))));
end
end