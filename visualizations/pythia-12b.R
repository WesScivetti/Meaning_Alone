library(tidyverse)
library(scales)
library(ggtext)

evals_raw <- read_tsv("accuracy_summary_la.tsv")

syntax_raw <- read_tsv("accuracy_summary_syntax.tsv")

syntax <- syntax_raw %>%
  filter(test_type == "Avg_Test") %>%
  mutate(
    step = str_remove(revision, "step") %>%
      as.numeric()
  ) %>%
  group_by(step) %>%
  summarize(accuracy = mean(accuracy)) %>%
  mutate(
    dataset = "PF Form"
  )


evals_raw %>%
  filter(!construction %in% c("blimp", "ewok", "comps")) %>%
  filter(!world_knowledge_filter %in% c("all", "aligned", "misaligned", "aligned1", "misaligned1", "aligned2", "misaligned2")) %>%
  filter(!str_detect(world_knowledge_filter, "_all")) %>%
  mutate(
    aligned = str_detect(world_knowledge_filter, "_aligned"),
    verb = str_extract(world_knowledge_filter, "(?<=verb_)(.*)(?=_(aligned|misaligned))"),
    step = str_remove(revision, "step") %>%
      as.numeric()
  ) %>%
  filter(aligned == T) %>%
  ggplot(aes(step, accuracy, color = verb)) +
  geom_line()+
  facet_wrap(~construction)

construction_combined <- evals_raw %>%
  filter(!construction %in% c("blimp", "ewok", "comps")) %>%
  mutate(
    construction = case_when(
      construction == "letalone" ~ "Let alone",
      construction == "muchless" ~ "Much less",
      construction == "nottomention" ~ "Not to mention",
      construction == "nevermind" ~ "Never mind",
      TRUE ~ construction
    ),
    step = str_remove(revision, "step") %>%
      as.numeric()
  ) %>%
  filter(world_knowledge_filter == "aligned")
  # filter(world_knowledge_filter %in% c("aligned", "misaligned"))

construction_best <- construction_combined %>%
  group_by(construction, world_knowledge_filter) %>%
  filter(step == max(step)) %>%
  ungroup()

construction_combined %>%
  filter(world_knowledge_filter == "aligned") %>%
  ggplot(aes(step, accuracy, color = construction, shape = construction, fill = construction)) +
  # geom_line(aes(linetype = world_knowledge_filter)) +
  geom_line() +
  geom_point(data = construction_best) +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  scale_shape_manual(values = c(21,22,23,24)) +
  scale_color_manual(values = c("#009392FF", "#EEB479FF", "#3299FFFF", "#CA562CFF"), aesthetics=c("color", "fill")) +
  scale_y_continuous(limits = c(0.25, 1), breaks = c(0.25, 0.5, 0.75, 1.0), labels = scales::percent_format(suffix = "")) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    # legend.spacing.y = unit(2.0, 'cm')
    legend.key.size = unit(1, "cm"),
    # legend.text = element_text(size = 12)
    legend.text = element_markdown(size = 12)
  ) +
  # guides(color = guide_legend(byrow = TRUE)) +
  labs(
    x = "Training Step",
    y = "Accuracy (%)",
    color = "Subset",
    shape = "Subset",
    fill = "Subset",
    # fill = "Dataset"
  )

ggsave("~/projects/plots-for-projects/accuracy-dynamics-perconstruction-pythia12b.pdf", height = 3.94, width = 6.17, dpi=300, device=cairo_pdf)

construction_combined %>%
  ggplot(aes(step, accuracy, color = construction, shape = construction, fill = construction)) +
  geom_line() +
  geom_point(data = construction_best) +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  scale_shape_manual(values = c(21,22,23,24)) +
  scale_y_continuous(limits = c(0.25, 1), breaks = c(0.25, 0.5, 0.75, 1.0), labels = scales::percent_format(suffix = "")) +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    # legend.spacing.y = unit(2.0, 'cm')
    legend.key.size = unit(1, "cm"),
    # legend.text = element_text(size = 12)
    legend.text = element_markdown(size = 12)
  ) +
  # guides(color = guide_legend(byrow = TRUE)) +
  labs(
    x = "Training Step",
    y = "Accuracy (%)",
    color = "Construction",
    shape = "Construction",
    fill = "Construction",
    # fill = "Dataset"
  )


  

evals_raw %>%
  filter(!construction %in% c("blimp", "ewok", "comps")) %>%
  filter(world_knowledge_filter == "aligned") %>%
  widyr::pairwise_cor(construction, revision, accuracy, diag=TRUE) %>%
  ggplot(aes(item1, item2, color = correlation, fill = correlation))+
  geom_tile() + 
  geom_text(aes(label = round(correlation, 2)), color = "white") +
  scale_color_distiller(direction = 1, aesthetics = c("color", "fill"), palette = "RdPu")

evals <- evals_raw %>%
  filter(world_knowledge_filter %in% c("aligned", "misaligned", "all")) %>%
  mutate(
    dataset = case_when(
      construction %in% c("blimp", "ewok", "comps") ~ construction,
      TRUE ~ glue::glue("PF Function\n({world_knowledge_filter})"),
    ),
    dataset = case_when(
      construction == "blimp" ~ "BLIMP",
      construction == "ewok" ~ "EWoK",
      construction == "comps" ~ "COMPS",
      TRUE ~ dataset
    ),
    step = str_remove(revision, "step") %>%
      as.numeric()
  ) %>%
  filter(dataset != "PF Function\n(all)") %>%
  group_by(dataset, step) %>%
  summarize(
    accuracy = mean(accuracy)
  ) %>%
  ungroup()

combined <- bind_rows(
  evals,
  syntax
) %>% mutate(
  dataset = case_when(
    dataset == "PF Function\n(aligned)" ~ "PF Function<br><span style='font-size: 10pt;'>(<i>plausible</i>)</span>",
    dataset == "PF Function\n(misaligned)" ~ "PF Function<br><span style='font-size: 10pt;'>(<i>implausible</i>)</span>",
    TRUE ~ dataset
  ),
  dataset = factor(dataset, levels = c("BLIMP", "COMPS", "EWoK", "PF Form", "PF Function<br><span style='font-size: 10pt;'>(<i>plausible</i>)</span>", "PF Function<br><span style='font-size: 10pt;'>(<i>implausible</i>)</span>"))
)

best <- combined %>%
  group_by(dataset) %>%
  filter(step == max(step)) %>%
  ungroup()

combined %>%
  ggplot(aes(step, accuracy, color = dataset, shape = dataset, fill = dataset)) +
  geom_line() +
  geom_point(data = best) +
  # geom_point() +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  geom_hline(yintercept = 0.25, linetype = "dotted") +
  annotate(
    "text",
    x = 125500, y = 0.21,
    # label = "Chance (EWoK)",
    label = expression(italic('Chance ') (EWoK)),
    fontface = "italic",
    size = 4,
    parse = TRUE
  ) +
  geom_curve(
    x = 102000, y = 0.21,
    xend = 90000, yend = 0.24,
    color = "black",
    curvature = -0.25,
    arrow = arrow(length = unit(2, "mm")),
    linewidth = 0.2
  ) +
  annotate(
    "text",
    x = 124500, y = 0.42,
    # label = "Chance (EWoK)",
    label = expression(italic('Chance ') (Others)),
    fontface = "italic",
    size = 4,
    parse = TRUE
  ) +
  geom_curve(
    # x = 97000, y = 0.45,
    # xend = 85000, yend = 0.49,
    x = 101000, y = 0.42,
    xend = 88000, yend = 0.495,
    color = "black",
    curvature = -0.30,
    arrow = arrow(length = unit(2, "mm")),
    linewidth = 0.2
  ) +
  # geom_richtext(
  #   x = 125000, y=0.2, label = "<i>Chance</i> (EWoK)", color = "black", fill = NA,size = 3,
  #   family = "Times"
  # ) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(suffix = "")) +
  # scale_x_continuous(limits = c(0, ))
  # scale_x_log10() +
  # scale_x_log10(
  #   breaks = trans_breaks("log10", function(x) 10^x),
  #   labels = trans_format("log10", math_format(10^.x))
  # ) +
  # scale_color_brewer(palette = "Dark2") +
  scale_color_manual(values = c("#648FFF","#DC267F", "#FE6100", "#FFB000", "#66a61e", "#a6761d"), aesthetics = c("color", "fill")) +
  # scale_color_manual(values = c("#648FFF","#DC267F", "#FE6100", "#FFB000", "#66a61e", "grey40"), aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(21,22,23,24,25,8)) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    # legend.spacing.y = unit(2.0, 'cm')
    legend.key.size = unit(1, "cm"),
    # legend.text = element_text(size = 12)
    legend.text = element_markdown(size = 12)
  ) +
  # guides(color = guide_legend(byrow = TRUE)) +
  labs(
    x = "Training Step",
    y = "Accuracy(%)",
    color = "Dataset",
    shape = "Dataset",
    fill = "Dataset"
  )

ggsave("~/projects/plots-for-projects/accuracy-dynamics-pythia.pdf", height = 3.94, width = 6.17, dpi=300, device=cairo_pdf)


# log scale for illusion of phase transition


datasets_to_plot <- c(
  "PF Form"
)

combined_subset <- combined %>%
  filter(dataset %in% datasets_to_plot)

best_subset <- best %>%
  filter(dataset %in% datasets_to_plot)

combined_subset %>%
  ggplot(aes(step, accuracy, color = dataset, shape = dataset, fill = dataset)) +
  geom_line() +
  geom_point(data = best_subset) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(suffix = "")) +
  scale_color_manual(
    values = c(
      "BLIMP" = "#648FFF",
      "COMPS" = "#DC267F",
      "EWoK" = "#FE6100",
      "PF Form" = "#FFB000",
      "PF Function<br><span style='font-size: 10pt;'>(<i>plausible</i>)</span>" = "#66a61e",
      "PF Function<br><span style='font-size: 10pt;'>(<i>implausible</i>)</span>" = "#a6761d"
    ),
    aesthetics = c("color", "fill")
  ) +
  scale_shape_manual(
    values = c(
      "BLIMP" = 21,
      "COMPS" = 22,
      "EWoK" = 23,
      "PF Form" = 24,
      "PF Function<br><span style='font-size: 10pt;'>(<i>plausible</i>)</span>" = 25,
      "PF Function<br><span style='font-size: 10pt;'>(<i>implausible</i>)</span>" = 8
    )
  ) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    legend.key.size = unit(1, "cm"),
    legend.text = element_markdown(size = 12)
  ) +
  labs(
    x = "Training Step",
    y = "Accuracy(%)",
    color = "Dataset",
    shape = "Dataset",
    fill = "Dataset"
  )

ggsave("accuracy-dynamics-0.pdf", height = 3.94, width = 6.17, dpi=300, device=cairo_pdf)

#######


combined %>%
  ggplot(aes(step, accuracy, color = dataset, shape = dataset, fill = dataset)) +
  geom_line() +
  geom_point(data = best) +
  # geom_point() +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  geom_hline(yintercept = 0.25, linetype = "dotted") +
  annotate(
    "text",
    x = 55, y = 0.20,
    # label = "Chance (EWoK)",
    label = expression(italic('Chance ') (EWoK)),
    fontface = "italic",
    size = 4,
    parse = TRUE
  ) +
  geom_curve(
    x = 0.8, y = 0.20,
    xend = 0.5, yend = 0.25,
    color = "black",
    curvature = -0.2,
    arrow = arrow(length = unit(2, "mm")),
    linewidth = 0.2
  ) +
  annotate(
    "text",
    x = 5, y = 0.64,
    # label = "Chance (EWoK)",
    label = expression(italic('Chance ') (Others)),
    fontface = "italic",
    size = 4,
    parse = TRUE
  ) +
  geom_curve(
    # x = 97000, y = 0.45,
    # xend = 85000, yend = 0.49,
    x = 0.5, y = 0.60,
    xend = 0.6, yend = 0.51,
    color = "black",
    curvature = 0.2,
    arrow = arrow(length = unit(2, "mm")),
    linewidth = 0.2
  ) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(suffix = "")) +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_color_manual(values = c("#648FFF","#DC267F", "#FE6100", "#FFB000", "#66a61e", "#a6761d"), aesthetics = c("color", "fill")) +
  scale_shape_manual(values = c(21,22,23,24,25,8)) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    legend.key.size = unit(1, "cm"),
    legend.text = element_markdown(size = 12)
  ) +
  labs(
    x = "Training Step (log-transformed)",
    y = "Accuracy(%)",
    color = "Dataset",
    shape = "Dataset",
    fill = "Dataset"
  )

ggsave("~/projects/plots-for-projects/accuracy-dynamics-pythia-logscale.pdf", height = 3.94, width = 6.17, dpi=300, device=cairo_pdf)


