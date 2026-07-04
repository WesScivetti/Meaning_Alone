library(tidyverse)
library(ggtext)

evals_raw <- read_tsv("../outputs_summary/jhu-clsp/ettin-decoder-1b/accuracy_summary_la_decoder1b.tsv")
syntax_raw <- read_tsv("../outputs_summary/jhu-clsp/ettin-decoder-1b/accuracy_summary_syntax_decoder1b.tsv")

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
  # filter(step %in% c(0, 2999, 8994, 26982, 80935, 149879)) %>%
  # filter(step %in% c(0, 2999, 5996, 8994, 17988, 32976, 65948, 128895, 149879)) %>%
  ggplot(aes(step, accuracy, color = dataset, shape = dataset, fill = dataset)) +
  geom_line() +
  geom_point(data = best) +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  geom_hline(yintercept = 0.25, linetype = "dotted") +
  annotate(
    "text",
    x = 128000, y = 0.21,
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
    x = 132000, y = 0.45,
    # label = "Chance (EWoK)",
    label = expression(italic('Chance ') (Others)),
    fontface = "italic",
    size = 4,
    parse = TRUE
  ) +
  geom_curve(
    # x = 97000, y = 0.45,
    # xend = 85000, yend = 0.49,
    x = 106000, y = 0.45,
    xend = 96000, yend = 0.495,
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
    x = "Step",
    y = "Accuracy(%)",
    color = "Dataset",
    shape = "Dataset",
    fill = "Dataset"
  )

ggsave("~/projects/plots-for-projects/accuracy-dynamics-ettin-decoder1b.pdf", height = 3.94, width = 6.17, dpi=300, device=cairo_pdf)
