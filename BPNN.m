
%% 1.��ʼ��
clear
close all
clc
format short %��ȷ��С�����4λ��format long�Ǿ�ȷ��С�����15λ
%% 2.��ȡ����
% �������ݼ�˳��
data = data(randperm(size(data, 1)), :);

% �������������������
input=data(:,1:end-1);    %��1����������2��Ϊ����
output=data(:,end);       %���1��Ϊ���

N=length(output);       %������������
testNum= length(output) *0.25 ;           %�趨���Լ����������������ݼ�����ѡȡ  
trainNum=N-testNum;     %�趨ѵ������������
%% 3.����ѵ�����Ͳ��Լ�
input_train = input(1:trainNum,:)';                   % ѵ��������
output_train =output(1:trainNum)';                    % ѵ�������
input_test =input(trainNum+1:trainNum+testNum,:)';    % ���Լ�����
output_test =output(trainNum+1:trainNum+testNum)';    % ���Լ����
%% 4.���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);         % ѵ���������һ����[0,1]֮�䣬���ٲ�ͬ���������Ĳ���
[outputn,outputps]=mapminmax(output_train);          % ѵ���������һ����Ĭ������[-1, 1]���������ȡֵ��Χ
inputn_test=mapminmax('apply',input_test,inputps);   % ���Լ�������ú�ѵ����������ͬ�Ĺ�һ����ʽ
%% 5.������������
inputnum=size(input,2);   %size������ȡ�����������������1����������2��������
outputnum=size(output,2); 
disp(['�����ڵ�����',num2str(inputnum),',  �����ڵ�����',num2str(outputnum)]) %num2str��������ת��Ϊ�ַ�������
disp(['������ڵ�����ΧΪ ',num2str(fix(sqrt(inputnum+outputnum))+1),' �� ',num2str(fix(sqrt(inputnum+outputnum))+10)])
disp(' ')
disp('���������ڵ��ȷ��...')
 
%����hiddennum=sqrt(m+n)+a��mΪ�����ڵ�����nΪ�����ڵ�����aȡֵ[1,10]֮�������
MSE=1e+5;                             %����ʼ��
transform_func={'tansig','purelin'};  %���������tan-sigmoid��purelin
train_func='trainlm';                 %ѵ���㷨�������Ż��㷨������Ѱ�������С
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    net=newff(inputn,outputn,hiddennum,transform_func,train_func); %����BP����
    
    % �����������
    net.trainParam.epochs=1000;       % ����ѵ������
    net.trainParam.lr=0.00001;           % ����ѧϰ����
    %net.trainParam.goal=1e-6;     % ����ѵ��Ŀ����С���
    net.trainParam.max_fail=60;      % ��Сȷ��ʧ�ܴ��� 
    
    % ��������ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);     %������
    mse0=mse(outputn,an0);   %����ľ������
    disp(['��������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����������Ϊ��',num2str(mse0)])
    
    %���ϸ������������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['���������ڵ���Ϊ��',num2str(hiddennum_best),'���������Ϊ��',num2str(MSE)])
%% 6.��������������BP������
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func);

% �������
net.trainParam.epochs=1000;         % ѵ������
net.trainParam.lr=0.00001;             % ѧϰ����
%net.trainParam.goal=1e-6;       % ѵ��Ŀ����С���
net.trainParam.max_fail=60;
%% 7.����ѵ��
net=train(net,inputn,outputn);      % train��������ѵ�������磬������ɫ�������
%% 8.�������
an=sim(net,inputn_test);                     % ѵ����ɵ�ģ�ͽ��з������
test_simu=mapminmax('reverse',an,outputps);  % ���Խ������һ��
error=test_simu-output_test;                 % ����ֵ����ʵֵ�����

% Ȩֵ��ֵ
W1 = net.iw{1, 1};  %����㵽�м���Ȩֵ
B1 = net.b{1};      %�м������Ԫ��ֵ
W2 = net.lw{2,1};   %�м�㵽������Ȩֵ
B2 = net.b{2};      %��������Ԫ��ֵ
%% 9.������
% BPԤ��ֵ��ʵ��ֵ�ĶԱ�ͼ
figure
plot(output_test,'bo-','linewidth',1.5)
hold on
plot(test_simu,'rs-','linewidth',1.5)
legend('ʵ��ֵ','Ԥ��ֵ')
xlabel('��������'),ylabel('ָ��ֵ')
title('BPԤ��ֵ��ʵ��ֵ�ĶԱ�')
set(gca,'fontsize',12)

% BP���Լ���Ԥ�����ͼ
figure
plot(error,'bo-','linewidth',1.5)
xlabel('��������'),ylabel('Ԥ�����')
title('BP��������Լ���Ԥ�����')
set(gca,'fontsize',12)

%�������������
[~,len]=size(output_test);            % len��ȡ����������������ֵ����testNum���������ָ��ƽ��ֵ
SSE1=sum(error.^2);                   % ���ƽ����
MAE1=sum(abs(error))/len;             % ƽ���������
MSE1=error*error'/len;                % �������
RMSE1=MSE1^(1/2);                     % ���������
MAPE1=mean(abs(error./output_test));  % ƽ���ٷֱ����
r=corrcoef(output_test,test_simu);    % corrcoef�������ϵ�����󣬰�������غͻ����ϵ��
R1=r(1,2);    

% ��ʾ��ָ����
disp(' ')
disp('�������ָ������')
disp(['���ƽ����SSE��',num2str(SSE1)])
disp(['ƽ���������MAE��',num2str(MAE1)])
disp(['�������MSE��',num2str(MSE1)])
disp(['���������RMSE��',num2str(RMSE1)])
disp(['ƽ���ٷֱ����MAPE��',num2str(MAPE1*100),'%'])
disp(['Ԥ��׼ȷ��Ϊ��',num2str(100-MAPE1*100),'%'])
disp(['���ϵ��R�� ',num2str(R1)])

%��ʾ���Լ����
disp(' ')
disp('���Լ������')
disp('    ���     ʵ��ֵ     BPԤ��ֵ     ���')
for i=1:len
    disp([i,output_test(i),test_simu(i),error(i)])   % ��ʾ˳��: ������ţ�ʵ��ֵ��Ԥ��ֵ�����
end
%% ģ�ʹ洢ָ��
save('Netmodel.mat', 'net');
disp('ѵ���õ�������ģ���ѱ���Ϊ trained_model.mat �ļ���')
