def IoU(bbox1, bbox2):
    x_min1, x_max1, y_min1, y_max1 = bbox1
    x_min2, x_max2, y_min2, y_max2 = bbox2
    x_min = max(x_min1, x_min2)
    y_min = max(y_min1, y_min2)
    x_max = min(x_max1, x_max2)
    y_max = min(y_max1, y_max2)

    x = max(0, x_max - x_min)
    y = max(0, y_max - y_min)
    intersection = x * y

    union = (
        (x_max1 - x_min1) * (y_max1 - y_min1)
        + (x_max2 - x_min2) * (y_max2 - y_min2)
        - intersection
    )
    return intersection / union


def yolo_loss(pred, names, bboxes):
    s_width = WIDTH / S
    s_height = HEIGHT / S
    coord_loss = perimeter_loss = 0
    confidence_loss = classification_loss = 0
    noobj_loss = 0
    lmb_ccord = 5
    lmb_noobj = 0.5

    for name, bbox in zip(names, bboxes):
        xmin, xmax, ymin, ymax = bbox
        x_c = (xmin + xmax) / 2.0
        y_c = (ymin + ymax) / 2.0
        section_x = int(x_c / s_width)
        section_y = int(y_c / s_height)
        w_gt = (xmax - xmin) / s_width  # gt - groung truth
        h_gt = (ymax - ymin) / s_height

        # Cut out vector corresponding to the current cell
        pred_vector = pred[section_x, section_y]
        # print(pred_vector)
        ious = []
        for b in range(B):
            pred_bbox = pred_vector[b * 5 : (b + 1) * 5]
            ious.append(IoU(pred_bbox[:4], bbox))

        responsible_bbox = np.argmax(ious)
        x, y, w, h, c = pred_vector[responsible_bbox * 5 : (responsible_bbox + 1) * 5]
        w = torch.nn.functional.relu(w)
        h = torch.nn.functional.relu(h)
        pred_distribution = pred_vector[-C:]

        # Make ground truth [-0.5, 0.5] (relative to cell center)
        x_gt = (x_c - s_width * (section_x + 0.5)) / s_width
        y_gt = (y_c - s_height * (section_y + 0.5)) / s_height

        # Calculate loss for the responsible bbox
        coord_loss += (x - x_gt) ** 2 + (y - y_gt) ** 2
        perimeter_loss += (w ** 0.5 - w_gt ** 0.5) ** 2 + (h ** 0.5 - h_gt ** 0.5) ** 2
        confidence_loss += (c - ious[responsible_bbox]) ** 2
        distribution_gt = torch.zeros(C)
        distribution_gt[class_to_index[name[0]]] = 1
        classification_loss += nn.MSELoss()(distribution_gt, pred_distribution)

        for b, iou in enumerate(ious):
            if b != responsible_bbox:
                c = pred_vector[b * 5 + 4]
                noobj_loss += (c - iou) ** 2

    #  for empty boxes:
    loss = (
        lmb_ccord * (coord_loss + perimeter_loss)
        + confidence_loss
        + classification_loss
        + lmb_noobj * noobj_loss
    )
    return loss
