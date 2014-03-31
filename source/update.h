#ifndef _UPDATE_H_
#define _UPDATE_H_

#include "types.h"

void update_sgd(layer_t *layer);
void update_sgd_momentum(layer_t *layer);
void update_sgd_momentum_l2_decay(layer_t *layer);

#endif
