%DAE的主函数 
function []=DAE_main(B,CentralBent,CockedRotor,Combined,CoupleBent,EC,IR,N,OR,Ub)
%clc,clear;
%全局变量
global layers_num neurals_num trainscale testscale classifications ...
    class datatextname sample stimulate;

%导入配置
DAE_config();

%检查配置
if(layers_num ~= length(neurals_num))
    sprintf('输入有误' )
end

%开始初始化权重
for i=1:(layers_num-1)             %对一个layer_num层的模型有layer_num-1组权重
    W{1,i} = zeros(neurals_num(1,i+1),neurals_num(1,i));
    w=[];
    w(1:neurals_num(1,i+1),1:neurals_num(1,i)) = DAE_iniweight(neurals_num(1,i+1),neurals_num(1,i));

    W{1,i} = w(1:neurals_num(1,i+1),1:neurals_num(1,i));     %w为各层间权重矩阵 W为管理w的元胞数组
end
%初始化特征组合权重
combin_W = DAE_iniweight(classifications,neurals_num(1,layers_num));
%开始导入数据
[train_data,train_label,test_data,test_label] = DAE_dataprocess(B,CentralBent,CockedRotor,Combined,CoupleBent,EC,IR,N,OR,Ub);

%------------------------------------------------------------------------------------------
%开始训练
%
%------------------------------------------------------------------------------------------
%逐层训练过程（无监督训练过程）
%取出训练数据，AE_input用于记录每一层AE的输入
AE_input = train_data(:,:);
lambda = 0.1;
acc_err = 0;
non_supervision_err = 0;
for j = 1:(layers_num-1)      %对每一层AE
    W_temp = W{1,j}';
    lambda = 0.01;  %新的一层重置学习率
    
    %对每组训练数据
    for k = 1:100  

        all_error = 0;
%%        
        for i = 1:size(train_data,2)          
            front = AE_input(:,i);       %用于暂时存储当前AE的输入数据  
            behind = [];                   %用于暂时存储计算结果
            %计算可见层到隐藏层
            behind = W{1,j}*front;    %计算下层输入
            hide = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%计算隐藏层 
            if(hide(:,:) <= 0)
                %监测全零列  断点处=============================
%                 sprintf('隐藏层出现全零列在无监督，第%f轮，第%f个数据',k,i)
            end

            %计算隐藏层到输出层
            output = DAE_stimulate(stimulate{1,1},stimulate{1,2},W_temp*hide);
            if(hide(:,:) <= 0)
                %监测全零列  断点处=============================
%                 sprintf('输出层出现全零列在无监督，第%f轮，第%f个数据',k,i)
            end
            %计算误差
            [temp_error,temp_diff_error] = DAE_errorfunction(front,output,{W{1,j},W_temp});
            all_error = all_error + temp_error;           
            %计算解码权重梯度
            %==========================sigmoid================================
%             delta_W{1,2} = temp_diff_error.*(output.*(1-output))*hide';
            %================================================================
            
            %==========================ReLU================================
            temp_diff_error(output == 0) = 0;
            delta_W2 = temp_diff_error * hide';
            %==============================================================
            
            %==========================leakReLU================================
%             temp_diff_error(output <= 0) = 0.001*temp_diff_error(output <= 0);
%             delta_W2 = temp_diff_error * hide';
            %==============================================================

            %计算编码权重梯度
        %==========================sigmoid================================
%         hide_diff_error = [];%输出层传导到隐藏层的误差
%         for k = 1:size(W_temp,2)
%             hide_diff_error = [hide_diff_error sum(temp_diff_error.*W_temp(:,k))];
%         end
%         hide_diff_error = hide_diff_error';
%         delta_W{1,1} = (hide_diff_error.* (hide.*(1-hide))*AE_input(:,i)');

        %================================================================
        %==========================ReLU================================

        hide_diff_error = W_temp'* temp_diff_error;
        hide_diff_error(hide == 0) = 0;
        delta_W1 = hide_diff_error * front';

        %==================================================================
        
        %==========================leakReLU================================

%         hide_diff_error = W_temp'* temp_diff_error;
%         hide_diff_error(hide <= 0) = 0.001*hide_diff_error(hide <= 0);
%         delta_W1 = hide_diff_error * front';

        %==================================================================
        ran_lambda = 0.01;
        %全零列问题保护修正
        if(hide(:,:) <= 0)
            if(front(:,:)>0)
                delta_W1 = - ran_lambda *rand(size(hide_diff_error,1),size(front,1));
            else
                if(front(:,:)<0)
                    delta_W1 =  ran_lambda *rand(size(hide_diff_error,1),size(front,1));
                else
                    delta_W1 = ran_lambda *(rand(size(hide_diff_error,1),size(front,1)) - 0.5);
                end
            end
        end    
        if(output(:,:) <= 0)
            if(hide(:,:)>0)
                delta_W2 = - ran_lambda *rand(size(front,1),size(hide_diff_error,1));
            else
                if(hide(:,:)<0)
                    delta_W2 =  ran_lambda *rand(size(front,1),size(hide_diff_error,1));
                else
                    delta_W2 = ran_lambda *(rand(size(front,1),size(hide_diff_error,1)) - 0.5);
                end
            end
        end
        
        
            W_temp = W_temp - lambda*delta_W2;%优化解码权重
            W{1,j} = W{1,j} - lambda*delta_W1;  %优化编码权重
            temp_error = [];
            delta_W1 = [];
            delta_W2 = [];
        end
        non_supervision_err = all_error/(size(train_data,2)*size(AE_input,1))
        k = k
        acc_err = acc_err + all_error/(size(train_data,2)*size(AE_input,1));

        if(mod(k,10) == 0)
            if(acc_err/10 <= non_supervision_err*1.001)
                lambda = lambda/10;
                sprintf('学习率衰减到%f',lambda)
            end
            acc_err = 0;
        end
       if (all_error/(size(train_data,2)*size(AE_input,1))<0.001)
          break;
       end
    end
    %用优化完成的AE计算下层AE的输入
    temp = [];
     for i = 1:size(train_data,2)
         
        front = AE_input(:,i);
        temp =[temp DAE_stimulate(stimulate{1,1},stimulate{1,2},W{1,j}*front)];
     end
      AE_input = [];
      AE_input = temp;
end

%分类训练过程（有监督训练）
sup_lambda = 1;

for k = 1:100

    for i = 1:size(train_data,2)
        layerIO{1,1} = train_data(:,i);%layerIO用于存储各层神经元的输入输出，第一行表示输入，第二行表示输出   
        layerIO{2,1} = train_data(:,i);%每一列对应于各层神经元 
        front = train_data(:,i);
        %特征提取
        for j = 1:(layers_num-1) 
            layerIO{2,j} = front;      %记录第J层神经元输出
            behind = W{1,j}*front;     %计算下层输入
            layerIO{1,j+1} = behind;   %记录第J+1层神经元输入
            front = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%计算下一层输出
            if(front(:,1) == 0)
                %监测全零列  断点处=============================
%                 sprintf('出现全零列在有监督训练的无监督计算过程中，第%f轮，第%f个数据，第%f层',k,i,j)
            end
        end 
        %记录DAE网络的输出
        layerIO{2,layers_num} = behind;
        %=================================保存无监督特征向量===============================
        Unsupervised_feature(:,i) = behind; 
        %=================================================================================
        if(layerIO{2,layers_num}(:,1) == 0)
            %监测全零列  断点处=============================
%             sprintf('出现全零列在有监督训练的无监督输出，第%f轮，第%f个数据',k,i)
        end
        %特征组合
        feature_combination = combin_W * layerIO{2,layers_num};
%         feature_combination = DAE_stimulate(stimulate{1,1},stimulate{1,2},combin_W * layerIO{2,layers_num}); 
        %分类器分类   
        [classifier,errrrrr,diff_classifier ]= DAE_classifier(feature_combination,train_label(:,i));
        errrrrr = errrrrr
        %分类误差

        delta_W1 = zeros(classifications,size(classifier,1));
        delta_W1 = diff_classifier * layerIO{2,layers_num}';
 
        %更新组合权重
        [~,indexx] = max(classifier);%预测结果
        [~,indexy] = max(train_label( : ,  i ));%真实结果
        if(indexx == indexy)
            sup_lambda = 3; %训练时，结果正确，修正衰减
        else
            sup_lambda = 30;%训练时，结果错误，修正增大
        end

        combin_W = combin_W - 0.01 * sup_lambda * delta_W1;
        delta_W1 = [];
    end
end

        %=================================输出无监督特征向量===============================

        %=================================================================================
%------------------------------------------------------------------------------------------
%开始测试
%
%------------------------------------------------------------------------------------------
forecast = [];
for i = 1:size(test_data,2)      %对每组训练数据
    front = test_data(:,i);     
    behind = [];
    
    %前向过程
    for j = 1:(layers_num-1)      %对每一层权值连接
       behind = W{1,j}*front;    %计算下层输入              
       front = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%计算J+1层神经元输出    
    end
    %front = DAE_stimulate('sigmoid',stimulate{1,2},front);
    %特征组合
    feature_combination = DAE_stimulate(stimulate{1,1},stimulate{1,2},combin_W * behind); 
    %分类器分类   
    [classification,~ ,~]= DAE_classifier(feature_combination,test_label(:,i));   
    %记录分类结果
    forecast = [forecast classification];

end
     forecast = forecast
hitNum(1,1:classifications) = 0;
mistakes(1,1:classifications) = 0;
 j = 0;
 %下面计算预测准确率
 %由于在dataprocess中各种类型数据是依次排列的，所以这里各种类型也是轮番测试。

for i = 1:size(forecast,2)
     [m , Index] = max(  forecast( : ,  i ) ) ;    %选出预测结果中最大值对应位置
     [n , Indexn] = max( test_label( : ,  i ) );   %选出标签对应位置
     
     if( Index  == Indexn) 
          hitNum(1,Indexn) = hitNum(1,Indexn) + 1 ; 
     else
         mistakes(1,Index) = mistakes(1,Index) + 1 ;
     end
end

%===================================================================================
%显示结果
%===================================================================================

for k = 1:1:classifications
    lvl = 100*hitNum(1,k)/testscale(1,1);
    string = class{1,k};
    sprintf('%s识别率是 %3.3f%% 误判%i',string,lvl,mistakes(1,k))
end
     totolhit = sum(hitNum,2);
sprintf('总识别率是 %3.3f%%',100 * totolhit / (testscale(1,1)*classifications) )

