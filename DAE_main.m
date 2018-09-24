%DAE�������� 
function []=DAE_main(B,CentralBent,CockedRotor,Combined,CoupleBent,EC,IR,N,OR,Ub)
%clc,clear;
%ȫ�ֱ���
global layers_num neurals_num trainscale testscale classifications ...
    class datatextname sample stimulate;

%��������
DAE_config();

%�������
if(layers_num ~= length(neurals_num))
    sprintf('��������' )
end

%��ʼ��ʼ��Ȩ��
for i=1:(layers_num-1)             %��һ��layer_num���ģ����layer_num-1��Ȩ��
    W{1,i} = zeros(neurals_num(1,i+1),neurals_num(1,i));
    w=[];
    w(1:neurals_num(1,i+1),1:neurals_num(1,i)) = DAE_iniweight(neurals_num(1,i+1),neurals_num(1,i));

    W{1,i} = w(1:neurals_num(1,i+1),1:neurals_num(1,i));     %wΪ�����Ȩ�ؾ��� WΪ����w��Ԫ������
end
%��ʼ���������Ȩ��
combin_W = DAE_iniweight(classifications,neurals_num(1,layers_num));
%��ʼ��������
[train_data,train_label,test_data,test_label] = DAE_dataprocess(B,CentralBent,CockedRotor,Combined,CoupleBent,EC,IR,N,OR,Ub);

%------------------------------------------------------------------------------------------
%��ʼѵ��
%
%------------------------------------------------------------------------------------------
%���ѵ�����̣��޼ලѵ�����̣�
%ȡ��ѵ�����ݣ�AE_input���ڼ�¼ÿһ��AE������
AE_input = train_data(:,:);
lambda = 0.1;
acc_err = 0;
non_supervision_err = 0;
for j = 1:(layers_num-1)      %��ÿһ��AE
    W_temp = W{1,j}';
    lambda = 0.01;  %�µ�һ������ѧϰ��
    
    %��ÿ��ѵ������
    for k = 1:100  

        all_error = 0;
%%        
        for i = 1:size(train_data,2)          
            front = AE_input(:,i);       %������ʱ�洢��ǰAE����������  
            behind = [];                   %������ʱ�洢������
            %����ɼ��㵽���ز�
            behind = W{1,j}*front;    %�����²�����
            hide = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%�������ز� 
            if(hide(:,:) <= 0)
                %���ȫ����  �ϵ㴦=============================
%                 sprintf('���ز����ȫ�������޼ල����%f�֣���%f������',k,i)
            end

            %�������ز㵽�����
            output = DAE_stimulate(stimulate{1,1},stimulate{1,2},W_temp*hide);
            if(hide(:,:) <= 0)
                %���ȫ����  �ϵ㴦=============================
%                 sprintf('��������ȫ�������޼ල����%f�֣���%f������',k,i)
            end
            %�������
            [temp_error,temp_diff_error] = DAE_errorfunction(front,output,{W{1,j},W_temp});
            all_error = all_error + temp_error;           
            %�������Ȩ���ݶ�
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

            %�������Ȩ���ݶ�
        %==========================sigmoid================================
%         hide_diff_error = [];%����㴫�������ز�����
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
        %ȫ�������Ᵽ������
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
        
        
            W_temp = W_temp - lambda*delta_W2;%�Ż�����Ȩ��
            W{1,j} = W{1,j} - lambda*delta_W1;  %�Ż�����Ȩ��
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
                sprintf('ѧϰ��˥����%f',lambda)
            end
            acc_err = 0;
        end
       if (all_error/(size(train_data,2)*size(AE_input,1))<0.001)
          break;
       end
    end
    %���Ż���ɵ�AE�����²�AE������
    temp = [];
     for i = 1:size(train_data,2)
         
        front = AE_input(:,i);
        temp =[temp DAE_stimulate(stimulate{1,1},stimulate{1,2},W{1,j}*front)];
     end
      AE_input = [];
      AE_input = temp;
end

%����ѵ�����̣��мලѵ����
sup_lambda = 1;

for k = 1:100

    for i = 1:size(train_data,2)
        layerIO{1,1} = train_data(:,i);%layerIO���ڴ洢������Ԫ�������������һ�б�ʾ���룬�ڶ��б�ʾ���   
        layerIO{2,1} = train_data(:,i);%ÿһ�ж�Ӧ�ڸ�����Ԫ 
        front = train_data(:,i);
        %������ȡ
        for j = 1:(layers_num-1) 
            layerIO{2,j} = front;      %��¼��J����Ԫ���
            behind = W{1,j}*front;     %�����²�����
            layerIO{1,j+1} = behind;   %��¼��J+1����Ԫ����
            front = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%������һ�����
            if(front(:,1) == 0)
                %���ȫ����  �ϵ㴦=============================
%                 sprintf('����ȫ�������мලѵ�����޼ල��������У���%f�֣���%f�����ݣ���%f��',k,i,j)
            end
        end 
        %��¼DAE��������
        layerIO{2,layers_num} = behind;
        %=================================�����޼ල��������===============================
        Unsupervised_feature(:,i) = behind; 
        %=================================================================================
        if(layerIO{2,layers_num}(:,1) == 0)
            %���ȫ����  �ϵ㴦=============================
%             sprintf('����ȫ�������мලѵ�����޼ල�������%f�֣���%f������',k,i)
        end
        %�������
        feature_combination = combin_W * layerIO{2,layers_num};
%         feature_combination = DAE_stimulate(stimulate{1,1},stimulate{1,2},combin_W * layerIO{2,layers_num}); 
        %����������   
        [classifier,errrrrr,diff_classifier ]= DAE_classifier(feature_combination,train_label(:,i));
        errrrrr = errrrrr
        %�������

        delta_W1 = zeros(classifications,size(classifier,1));
        delta_W1 = diff_classifier * layerIO{2,layers_num}';
 
        %�������Ȩ��
        [~,indexx] = max(classifier);%Ԥ����
        [~,indexy] = max(train_label( : ,  i ));%��ʵ���
        if(indexx == indexy)
            sup_lambda = 3; %ѵ��ʱ�������ȷ������˥��
        else
            sup_lambda = 30;%ѵ��ʱ�����������������
        end

        combin_W = combin_W - 0.01 * sup_lambda * delta_W1;
        delta_W1 = [];
    end
end

        %=================================����޼ල��������===============================

        %=================================================================================
%------------------------------------------------------------------------------------------
%��ʼ����
%
%------------------------------------------------------------------------------------------
forecast = [];
for i = 1:size(test_data,2)      %��ÿ��ѵ������
    front = test_data(:,i);     
    behind = [];
    
    %ǰ�����
    for j = 1:(layers_num-1)      %��ÿһ��Ȩֵ����
       behind = W{1,j}*front;    %�����²�����              
       front = DAE_stimulate(stimulate{1,1},stimulate{1,2},behind);%����J+1����Ԫ���    
    end
    %front = DAE_stimulate('sigmoid',stimulate{1,2},front);
    %�������
    feature_combination = DAE_stimulate(stimulate{1,1},stimulate{1,2},combin_W * behind); 
    %����������   
    [classification,~ ,~]= DAE_classifier(feature_combination,test_label(:,i));   
    %��¼������
    forecast = [forecast classification];

end
     forecast = forecast
hitNum(1,1:classifications) = 0;
mistakes(1,1:classifications) = 0;
 j = 0;
 %�������Ԥ��׼ȷ��
 %������dataprocess�и��������������������еģ����������������Ҳ���ַ����ԡ�

for i = 1:size(forecast,2)
     [m , Index] = max(  forecast( : ,  i ) ) ;    %ѡ��Ԥ���������ֵ��Ӧλ��
     [n , Indexn] = max( test_label( : ,  i ) );   %ѡ����ǩ��Ӧλ��
     
     if( Index  == Indexn) 
          hitNum(1,Indexn) = hitNum(1,Indexn) + 1 ; 
     else
         mistakes(1,Index) = mistakes(1,Index) + 1 ;
     end
end

%===================================================================================
%��ʾ���
%===================================================================================

for k = 1:1:classifications
    lvl = 100*hitNum(1,k)/testscale(1,1);
    string = class{1,k};
    sprintf('%sʶ������ %3.3f%% ����%i',string,lvl,mistakes(1,k))
end
     totolhit = sum(hitNum,2);
sprintf('��ʶ������ %3.3f%%',100 * totolhit / (testscale(1,1)*classifications) )

