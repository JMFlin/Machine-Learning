###################################################################

## Slide 51 "Approximate Random Forest Resampled ROC Curve"

## This function averages the class probability values per sample
## across the hold-outs to get an averaged ROC curve

roc_train <- function(object, best_only = TRUE, ...) {
  
  
  lvs <- object$modelInfo$levels(object$finalModel)
  
  if(best_only) {
    object$pred <- merge(object$pred, object$bestTune)
  }
  
  ## find tuning parameter names
  p_names <- as.character(object$modelInfo$parameters$parameter)
  p_combos <- object$pred[, p_names, drop = FALSE]
  
  ## average probabilities across resamples
  object$pred <- ddply(.data = object$pred, #plyr::
                       .variables = c("obs", "rowIndex", p_names),
                       .fun = function(dat, lvls = lvs) {
                         out <- mean(dat[, lvls[1]])
                         names(out) <- lvls[1]
                         out
                       })
  
  make_roc <- function(x, lvls = lvs, nms = NULL, ...) {
    out <- roc(response = x$obs,#pROC::
               predictor = x[, lvls[1]],
               levels = rev(lvls))
    
    out$model_param <- x[1,nms,drop = FALSE]
    out
  }
  out <- plyr::dlply(.data = object$pred, 
                     .variables = p_names,
                     .fun = make_roc,
                     lvls = lvs,
                     nms = p_names)
  if(length(out) == 1)  out <- out[[1]]
  out
}

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")


## Slide 52 "A Better Cutoff"

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres.pattern = "Cutoff: %.2f (Sp = %.2f, Sn = %.2f)",
     print.thres = "best")
