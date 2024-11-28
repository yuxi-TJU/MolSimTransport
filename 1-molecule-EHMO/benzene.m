clear
clc

H = readmatrix('C:\Users\xuanj\Documents\works\TBTransmission\couplingsite\halfH-Hfirst.csv');
O = readmatrix('C:\Users\xuanj\Documents\works\TBTransmission\couplingsite\halfO-Hfirst.csv');
upperTriangle = triu(H);
upperTriangle_O = triu(O);

% 提取上三角部分的转置并去除对角线
lowerTriangle = tril(upperTriangle', -1);
lowerTriangle_O = tril(upperTriangle_O', -1);

% 将上三角和下三角部分相加以得到对称矩阵
HH = upperTriangle + lowerTriangle;
OO = upperTriangle_O + lowerTriangle_O;

[eigen_vector,eigen_value]=eig(HH,OO);
eigen_value1=eig(HH);


t=2;
N = length(HH);  
EEEE=linspace(0.5,0.5,N);
E_num = 10000;  
Erange = linspace(-10*t, 10*t, E_num);  
dE = abs(Erange(2) - Erange(1));  


Gamma1_wba=0.1;Gamma2_wba=Gamma1_wba;
Sigma1_wba = -0.5j*Gamma1_wba;Sigma2_wba = -0.5j*Gamma2_wba;
z_plus = 1e-13j;

for i=1:length(Erange)        
    Gamma1 = diag([zeros(1,6),Gamma1_wba,Gamma1_wba,Gamma1_wba,Gamma1_wba,zeros(1,20)]);
    Gamma2 = diag([zeros(1,18),Gamma2_wba,Gamma2_wba,Gamma2_wba,Gamma2_wba,zeros(1,8)]);
    Sigma1 = diag([zeros(1,6),Sigma1_wba,Sigma1_wba,Sigma1_wba,Sigma1_wba,zeros(1,20)]);
    Sigma2 = diag([zeros(1,18),Sigma2_wba,Sigma2_wba,Sigma2_wba,Sigma2_wba,zeros(1,8)]);

     G = inv(((Erange(i) + z_plus)*OO) - (HH + Sigma1 + Sigma2));  
          
      Tpara_list= Gamma1 * G * Gamma2 * conj(G);
      Tpara(i) = real(trace(Tpara_list));
end

  
for i=1:length(Erange)        
    Gamma1 = diag([zeros(1,6),Gamma1_wba,Gamma1_wba,Gamma1_wba,Gamma1_wba,zeros(1,20)]);
    Gamma2 = diag([zeros(1,14),Gamma2_wba,Gamma2_wba,Gamma2_wba,Gamma2_wba,zeros(1,12)]);
    Sigma1 = diag([zeros(1,6),Sigma1_wba,Sigma1_wba,Sigma1_wba,Sigma1_wba,zeros(1,20)]);
    Sigma2 = diag([zeros(1,14),Sigma2_wba,Sigma2_wba,Sigma2_wba,Sigma2_wba,zeros(1,12)]);

    G = inv(((Erange(i) + z_plus)*OO) - (HH + Sigma1 + Sigma2));   

    Tmeta_list = Gamma1 * G * Gamma2 * conj(G);
    Tmeta(i) = real(trace(Tmeta_list));
end  

TTpara=real(log10(Tpara));


plot(Erange,log10(Tmeta),'-','linewidth',1.5);hold on
plot(Erange,log10(Tpara),'--','linewidth',1.5);hold on
 scatter(eigen_value,EEEE,'MarkerEdgeColor','k','LineWidth',1,'SizeData',50)
title('From C 2s,p to C 2s,p')
xlabel('E(eV)')
ylabel('Transmission (log10)')
xlim([-15.0 -5.0])
legend('meta','para')