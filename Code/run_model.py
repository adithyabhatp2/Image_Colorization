import keras
import rzhang_model
import ImageUtils as myutils
from create_dataset import *
import cPickle


if __name__ == '__main__':

    keras.activations.softmax3 = rzhang_model.softmax3
    keras.losses.custom_loss = rzhang_model.custom_loss
    keras.losses.custom_loss_rebalancing = rzhang_model.custom_loss_rebalancing

    L_SIZE = 224
    AB_SIZE = 14

    model = rzhang_model.build_model()
    # model = keras.models.load_model('rzhang_model5.h5')

    # On bhat's google instance
    train_dir_path = '/home/adithya_bhatp/images/nature_hilly/train'
    val_dir_path = '/home/adithya_bhatp/images/nature_hilly/val'
    test_dir_path = '/home/adithya_bhatp/images/nature_hilly/test'
    test_dir_path = '/home/adithya_bhatp/images/nature_hilly/test_sample' # Make a sample dir with 10 imgs or something

    # myutils.cleanup_dir(dir_path)

    # train_l_channel, train_one_hot_ab = create_dataset(train_dir_path, L_SIZE, AB_SIZE)
    # Moved to create_dataset_v2. should be identical, just extra test_ab_bin_frequencies returned. ignores b&w input.
    train_l_channel, train_one_hot_ab, train_ab_bin_frequencies = create_dataset_v2(train_dir_path, L_SIZE, AB_SIZE)

    # Train
    NUM_EPOCHS = 50
    for i in range(NUM_EPOCHS):
        # model.fit(l_channel, one_hot_ab, batch_size = 32, epochs = 5, verbose = 1) - # srini 20170426
        # Can also give validation data here
        model.fit(train_l_channel, train_one_hot_ab, batch_size=32, epochs=1, verbose=1, shuffle=True)
        if i % 5 == 0:
            model.save('rzhang_model{0}.h5'.format(i))


    # # Load Test Data
    # test_l_channel, test_one_hot_ab_actual, train_ab_bin_frequencies = create_dataset_v2(train_dir_path, L_SIZE, AB_SIZE)
    # test_one_hot_abs_predicted = model.predict(test_l_channel)
    #
    # # Test
    # test_img_predictions_rgb = []
    # test_abs_predicted = decode_ab(test_one_hot_abs_predicted)
    # num_test_instances = test_l_channel.shape[0]
    #
    # for i in ():
    #     test_img_predicted_lab = myutils.merge_l_ab(test_l_channel[i], test_abs_predicted[i])
    #     test_img_predicted_rgb = myutils.convertLabToRgb(test_img_predicted_lab)
    #     # consider saving image. Also note that if we retain the original image size, we can resize to it.
    #     # hence leaving it as iterative.
    #     test_img_predictions_rgb.append(test_img_predicted_rgb)
    #
    # myutils.displayListOfImagesInGrid(test_img_predictions_rgb[0:16])
