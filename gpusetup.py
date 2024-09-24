import tensorflow as tf


def gpu_setup():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus[0].device_type:
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*2)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

        