
%% 1.初始化
clear
close all
clc
format short %精确到小数点后4位，format long是精确到小数点后15位
%% 2.读取数据
% 打乱数据集顺序
data = data(randperm(size(data, 1)), :); #请用matlab的导入数据功能导入数据

% 设置神经网络的输入和输出
input=data(:,1:end-1);    %第1列至倒数第2列为输入
output=data(:,end);       %最后1列为输出

N=length(output);       %计算样本数量
testNum= length(output) *0.3 ;           %设定测试集样本数量，从数据集后面选取  
trainNum=N-testNum;     %设定训练集样本数量
%% 3.设置训练集和测试集
input_train = input(1:trainNum,:)';                   % 训练集输入
output_train =output(1:trainNum)';                    % 训练集输出
input_test =input(trainNum+1:trainNum+testNum,:)';    % 测试集输入
output_test =output(trainNum+1:trainNum+testNum)';    % 测试集输出
%% 4.数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);         % 训练集输入归一化到[0,1]之间，减少不同量级带来的差异
[outputn,outputps]=mapminmax(output_train);          % 训练集输出归一化到默认区间[-1, 1]，激活函数的取值范围
inputn_test=mapminmax('apply',input_test,inputps);   % 测试集输入采用和训练集输入相同的归一化方式
%% 5.求解最佳隐含层
inputnum=size(input,2);   %size用来求取矩阵的行数和列数，1代表行数，2代表列数
outputnum=size(output,2); 
disp(['输入层节点数：',num2str(inputnum),',  输出层节点数：',num2str(outputnum)]) %num2str用来进行转换为字符串类型
disp(['隐含层节点数范围为 ',num2str(fix(sqrt(inputnum+outputnum))+1),' 至 ',num2str(fix(sqrt(inputnum+outputnum))+10)])
disp(' ')
disp('最佳隐含层节点的确定...')
 
%根据hiddennum=sqrt(m+n)+a，m为输入层节点数，n为输出层节点数，a取值[1,10]之间的整数
MSE=1e+5;                             %误差初始化
transform_func={'tansig','purelin'};  %激活函数采用tan-sigmoid和purelin
train_func='trainlm';                 %训练算法，采用优化算法，用于寻找误差最小
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    net=newff(inputn,outputn,hiddennum,transform_func,train_func); %构建BP网络
    
    % 设置网络参数
    net.trainParam.epochs=1000;       % 设置训练次数
    net.trainParam.lr=0.001;           % 设置学习速率
    net.trainParam.goal=1e-6;     % 设置训练目标最小误差
    net.trainParam.max_fail=60;      % 最小确认失败次数 
    
    % 进行网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);     %仿真结果
    mse0=mse(outputn,an0);   %仿真的均方误差
    disp(['当隐含层节点数为',num2str(hiddennum),'时，训练集均方误差为：',num2str(mse0)])
    
    %不断更新最佳隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳隐含层节点数为：',num2str(hiddennum_best),'，均方误差为：',num2str(MSE)])
%% 6.构建最佳隐含层的BP神经网络
net=newff(inputn,outputn,hiddennum_best,transform_func,train_func);

% 网络参数
net.trainParam.epochs=1000;         % 训练次数
net.trainParam.lr=0.001;             % 学习速率
net.trainParam.goal=1e-6;       % 训练目标最小误差
net.trainParam.max_fail=60;
%% 7.网络训练
net=train(net,inputn,outputn);      % train函数用于训练神经网络，调用蓝色仿真界面
%% 8.网络测试
an=sim(net,inputn_test);                     % 训练完成的模型进行仿真测试
test_simu=mapminmax('reverse',an,outputps);  % 测试结果反归一化
error=test_simu-output_test;                 % 测试值和真实值的误差

% 权值阈值
W1 = net.iw{1, 1};  %输入层到中间层的权值
B1 = net.b{1};      %中间各层神经元阈值
W2 = net.lw{2,1};   %中间层到输出层的权值
B2 = net.b{2};      %输出层各神经元阈值
%% 9.结果输出
% BP预测值和实际值的对比图
figure
plot(output_test,'bo-','linewidth',1.5)
hold on
plot(test_simu,'rs-','linewidth',1.5)
legend('实际值','预测值')
xlabel('测试样本'),ylabel('指标值')
title('BP预测值和实际值的对比')
set(gca,'fontsize',12)

% BP测试集的预测误差图
figure
plot(error,'bo-','linewidth',1.5)
xlabel('测试样本'),ylabel('预测误差')
title('BP神经网络测试集的预测误差')
set(gca,'fontsize',12)

%计算各项误差参数
[~,len]=size(output_test);            % len获取测试样本个数，数值等于testNum，用于求各指标平均值
SSE1=sum(error.^2);                   % 误差平方和
MAE1=sum(abs(error))/len;             % 平均绝对误差
MSE1=error*error'/len;                % 均方误差
RMSE1=MSE1^(1/2);                     % 均方根误差
MAPE1=mean(abs(error./output_test));  % 平均百分比误差
r=corrcoef(output_test,test_simu);    % corrcoef计算相关系数矩阵，包括自相关和互相关系数
R1=r(1,2);    

% 显示各指标结果
disp(' ')
disp('各项误差指标结果：')
disp(['误差平方和SSE：',num2str(SSE1)])
disp(['平均绝对误差MAE：',num2str(MAE1)])
disp(['均方误差MSE：',num2str(MSE1)])
disp(['均方根误差RMSE：',num2str(RMSE1)])
disp(['平均百分比误差MAPE：',num2str(MAPE1*100),'%'])
disp(['预测准确率为：',num2str(100-MAPE1*100),'%'])
disp(['相关系数R： ',num2str(R1)])

%显示测试集结果
disp(' ')
disp('测试集结果：')
disp('    编号     实际值     BP预测值     误差')
for i=1:len
    disp([i,output_test(i),test_simu(i),error(i)])   % 显示顺序: 样本编号，实际值，预测值，误差
end
%% 模型存储指令
save('Netmodel.mat', 'net');
disp('训练好的神经网络模型已保存为 trained_model.mat 文件。')
