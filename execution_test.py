from experiments.common.definitions import remat_data_dir
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.load_keras_model import get_keras_model
from remat.tensorflow2.extraction import dfgraph_from_keras
from experiments.common.execution_utils import random_batch
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.tensorflow2.execution import tfgraph_from_schedule
from remat.tensorflow2.tf_losses import categorical_cross_entropy
import tensorflow as tf
import numpy as np
import logging



def test_exec_vgg16_checkpointall():
    log_dir = remat_data_dir() / "test_execution"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gurobipy as _
    except ImportError as e:
        logging.exception(e)
        logging.warning("Continuing with tests, gurobi not installed")
        return
    #model = get_keras_model("VGG19")
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model = get_keras_model("VGG19", input_shape=x_train[0].shape)

    g = dfgraph_from_keras(model)
    render_dfgraph(g, log_dir / "test_exec")

    schedule = solve_checkpoint_all(g)
    assert schedule.feasible
    assert schedule.schedule_aux_data.cpu <= sum(g.cost_cpu.values())
    assert schedule.schedule_aux_data.activation_ram <= sum(g.cost_cpu.values())
    
    loss = categorical_cross_entropy
    checkmate_graph = tfgraph_from_schedule(model, g, schedule, loss=loss, debug=True)
   
    element_spec = train_ds.__iter__().__next__()

    test_batch, test_labels = element_spec[0], element_spec[1]
    test_labels = tf.cast(test_labels, tf.int32)
    test_labels = tf.squeeze(test_labels, [1])
    one_hot = tf.one_hot(test_labels, 10)
    with tf.GradientTape(persistent=True) as tape:
        tf_pred = model(test_batch)
        tf_loss = loss(tf_pred, one_hot)
    tf_grads = tape.gradient(tf_loss, model.trainable_variables)
    
    our_loss, our_grads = checkmate_graph(test_batch, one_hot)
    print(f"TF baseline loss = {tf_loss}")
    print(f"Checkmate loss = {our_loss}")
    
    logging.info(f"TF baseline loss = {tf_loss}")
    logging.info(f"Checkmate loss = {our_loss}")


if __name__ == "__main__":
    test_exec_vgg16_checkpointall()