import math

def train_model(hp, model, train_loader, writer, logger):
    model.net.train()
    model.sub_step = 0

    target, target_grad, planewaves, OF_factor= train_loader 
    if OF_factor.max() > 0:
        model.feed_data(GT = target, GT_grad = target_grad, planewaves = planewaves, OF_factor = OF_factor)
    else:
        model.feed_data(GT = target, GT_grad = target_grad, planewaves = planewaves)

    model.optimize_parameters()
    loss = model.log.loss_v
    loss_state = model.log.loss_v_state
    lr = model.log.lr
    model.step += 1
    model.sub_step += 1

    if logger is not None and (math.isnan(loss)):
        logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
        raise Exception("Loss exploded")

    if hp.log.summary_interval and model.step % hp.log.summary_interval == 0:
        if writer is not None:
            writer.train_logging(loss, loss_state, model.output, model.GT, model.GT_grad, lr, model.epoch)
        
        if logger is not None:
            logger.info("^_^ Epoch %d : Train Loss %.04f at step %d sub_step %d; lr %.06f ^_^" % (model.epoch + 1,
                                                                                                  loss,
                                                                                                  model.step,
                                                                                                  model.sub_step,
                                                                                                  lr,
                                                                                                  )
                        )
            logger.info("----> Train: {}".format(loss_state))
