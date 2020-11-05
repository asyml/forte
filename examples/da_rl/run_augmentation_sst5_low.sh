for t in {1..15}
do
        python3 -u main.py \
        --train_num_per_class 40 \
        --dev_num_per_class 2 \
        --classifier_lr 4e-5 \
        --epochs 5 \
        --min_epochs 0 \
        --batch_size 8 \
        --generator_lr 4e-5 \
        --classifier_pretrain_epochs 3 \
        --generator_pretrain_epochs 60 \
        --n_aug 4
done
