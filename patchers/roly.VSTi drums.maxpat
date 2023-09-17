{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 5,
			"revision" : 6,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 34.0, 87.0, 972.0, 779.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"color" : [ 0.701961, 0.701961, 0.701961, 0.0 ],
					"id" : "obj-20",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 5,
							"revision" : 6,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 59.0, 106.0, 640.0, 480.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 12.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial",
						"gridonopen" : 1,
						"gridsize" : [ 15.0, 15.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"visible" : 1,
						"boxes" : [ 							{
								"box" : 								{
									"id" : "obj-10",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 50.0, 100.0, 348.0, 22.0 ],
									"text" : "plug \"C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3\""
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-8",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 50.0, 127.232361508224471, 249.0, 22.0 ],
									"text" : "plug /Library/Audio/Plug-Ins/VST3/LABS.vst3"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-15",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "bang" ],
									"patching_rect" : [ 44.0, 40.0, 30.0, 30.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-19",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 44.0, 209.232360999999969, 30.0, 30.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-19", 0 ],
									"source" : [ "obj-10", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-10", 0 ],
									"order" : 1,
									"source" : [ "obj-15", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-8", 0 ],
									"order" : 0,
									"source" : [ "obj-15", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-19", 0 ],
									"source" : [ "obj-8", 0 ]
								}

							}
 ]
					}
,
					"patching_rect" : [ 37.0, 619.0, 61.0, 22.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p load-vst"
				}

			}
, 			{
				"box" : 				{
					"color" : [ 0.701961, 0.701961, 0.701961, 0.0 ],
					"id" : "obj-14",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 37.0, 595.0, 58.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-85",
					"linecount" : 6,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 4.0, 71.0, 511.0, 94.0 ],
					"text" : "You can use rolypoly~ in message_out mode, which outputs note-velocity pairs to be interpreted by any drum sampler or synthesizer. Unless sub-millisecond precision is important to you, this option is probably the most easy to plug-and-play.\n\nThis patch uses the free LABS Drums instrument (https://labs.spitfireaudio.com/drums) but you can go ahead and plug in your favourite VSTi.",
					"textcolor" : [ 0.426676, 0.426663, 0.42667, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Lato",
					"fontsize" : 48.0,
					"id" : "obj-6",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 4.0, 6.0, 472.0, 64.0 ],
					"text" : "rolypoly~"
				}

			}
, 			{
				"box" : 				{
					"args" : [ "@file", "jongly.aif", "@loop", 1, "@vol", 0 ],
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-2",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "demosound.maxpat",
					"numinlets" : 0,
					"numoutlets" : 1,
					"offset" : [ -4.0, -3.0 ],
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 472.0, 196.666667999999987, 219.0, 89.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-28",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "roly.browse.maxpat",
					"numinlets" : 0,
					"numoutlets" : 0,
					"offset" : [ 0.0, 0.0 ],
					"patching_rect" : [ 568.0, 607.0, 123.0, 234.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 2,
					"numoutlets" : 5,
					"orientation" : 1,
					"outlettype" : [ "signal", "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 112.0, 723.0, 136.0, 47.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_longname" : "live.gain~[3]",
							"parameter_mmax" : 6.0,
							"parameter_mmin" : -70.0,
							"parameter_shortname" : "drums",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "live.gain~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 112.0, 796.0, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 112.0, 521.567627000000016, 32.5, 23.0 ],
					"text" : "join"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-18",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 112.0, 491.567627000000016, 107.0, 23.0 ],
					"text" : "makenote 60 4n"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-35",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 415.600007116794586, 548.799988508224487, 64.0, 22.0 ],
					"text" : "Reverb $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 333.0, 548.799988508224487, 78.0, 22.0 ],
					"text" : "Dynamics $1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-31",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 245.0, 548.799988508224487, 85.0, 22.0 ],
					"text" : "Expression $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-11",
					"maxclass" : "newobj",
					"numinlets" : 7,
					"numoutlets" : 2,
					"outlettype" : [ "int", "" ],
					"patching_rect" : [ 112.0, 552.799996554851532, 108.0, 23.0 ],
					"text" : "midiformat"
				}

			}
, 			{
				"box" : 				{
					"autosave" : 1,
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"id" : "obj-12",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 8,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "signal", "", "list", "int", "", "", "" ],
					"patching_rect" : [ 112.0, 593.0, 359.0, 116.0 ],
					"save" : [ "#N", "vst~", "loaduniqueid", 0, "C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3", ";" ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_invisible" : 1,
							"parameter_longname" : "vst~",
							"parameter_shortname" : "vst~",
							"parameter_type" : 3
						}

					}
,
					"saved_object_attributes" : 					{
						"parameter_enable" : 1,
						"parameter_mappable" : 0
					}
,
					"snapshot" : 					{
						"filetype" : "C74Snapshot",
						"version" : 2,
						"minorversion" : 0,
						"name" : "snapshotlist",
						"origin" : "vst~",
						"type" : "list",
						"subtype" : "Undefined",
						"embed" : 1,
						"snapshot" : 						{
							"pluginname" : "LABS.vst3",
							"plugindisplayname" : "LABS",
							"pluginsavedname" : "/Volumes/C/Program",
							"pluginsaveduniqueid" : 0,
							"version" : 1,
							"isbank" : 0,
							"isbase64" : 1,
							"sliderorder" : [  ],
							"slidervisibility" : [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
							"blob" : "13241.VMjLg.6L...OVMEUy.Ea0cVZtMEcgQWY9vSRC8Vav8lak4Fc9jCN2TiKV0zQicUPt3hKl4hKt3BTt3hKt3hKLoGVzMGQt3BV3QlQIoGTtEjKt3BR1QEa2oFVtPDTAUkKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKtrFSzU0PIMERjYkPt3hc48zLvXTXlg0UYgWSWoUczX0SnQTZKYGRBgzZzDCV0EkUZQ2XV8DZTUTUFAiPNg1Mo8jY1MzTmkTLhkicSMUQQUETlgkUXM2ZFEFMvjFRDkzUiMWSsgjYyXEVyUkUOgFTpIFLvDiXn4hPhgGNFkELMYzXMgiQYsFLogjcHIDRwTEahk2ZwDFcvjFR2MiPLQGSogjYPcEVs0zUOgFQCwzctLDS14RZNQTRWM1bM0FRloWLgo1Zrk0aUYTV3fjPLg1Mn8zMTUkTlQ0UZk2ZrQ1ZvjFR2MiPLglKRM1aMESXxcmUXYWSWkkZvjFR2gDdKkicSAkTQUkTC0zZOcCSUEEUQUkTNMFQH8VTV8DZtHyU4sVagkVTvDFUUYUX1gCaHYFVWgkbUcUV3fjTLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWo1ZsE1YvXkVo0TaUs1cwDVZqYzXz.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.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.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbEdUw1XqkTaXglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHcmZogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbkdqESVtEUaHYFVWgkbUcUV3fjPLQGUogjYLECV3fjTLMCRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNvHlcUYUVpkjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUVRWkkbUYEV4UEaHYFVWgkbUcUV3fjPLQGUogjYLECV3fjTLICRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUk0LA0lXq0jLh8FNrEFZtf1XmcmUisFLogzcyHDSn4BdXkFLogzcDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUpsVagcFLVoUZM0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHIESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkk0UXg2ZVgkdqESXzkjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SngzPLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUVVWoEZIcEV5gCaHYFVWgkbUcUV3fjTLQmKogjYLECV3fDZLcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUkEcYcEV5EUaHYFVWgkbUcUV3fjPLQmKS4DMpMkSzn1PNECUC4zLpMUS5oVdLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoVLWsFMrMlZUECVn4BZic1cVM1ZvjFR1MiPLQiZS4DMpMkSz.0PLkmZS0TdTMUS3gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkUEagESSWMVdIIDRwTjQgASUV8DZDk1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkUEagESRWkkbIIDRwTjQgASUV8DZHk1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkAiUZMSSWo0bAcTXqkjPHESQFEFLUY0Sn4RZKACRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fDdhUVVFE1aA0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFR4gCLi8VTFMlaIIDRwTjQgASUV8DZtj1RvfjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHgmXkEzUXQWRBgTLEYTXvTkUOglKosDLHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUYQYUVxUjUjglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwb0YvDSXvPiQiglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbkZqEiX5gCahoWRBgTLEYTXvTkUOglKosjcHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUY2Y0X4cFaUsVRsgEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU1UkQYECNFEFZtf1XmcmUisFLogjcyHUSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU1UkQYQzZsEFZtf1XmcmUisFLogjcyHUSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUxUEag0VTGoEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU5slUgsFLTgUZmYkVzUEaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNvHldIcUV50jQZglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbUd3vVV5ETUYoVQFEFZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU3UULhYGNrEVdUwFRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugSUgc1cFE1ZQ0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugCLhoGNFI1ZvP0X5UEaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNEk0aIcUVoE0UZUGMrgjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogza3TTX1kEUZIWTWkEdIIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkEjLggWTWg0bUwVX5gCaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUE1ZQcEVgUzPNMVRBgTLEYTXvTkUOglKosjcHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUYvXUV5UTLVcmZScEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUyUkQic1bqwjc5sFRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugSUgsVTWgUXIMESikjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUFLVkkdEEiU3gzTWglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwb0bUYzXmM2ZLkmdqgjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogza3TUXqE0UXEVRC0zXIIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkAiUYoWQwXEdTM0Un4BZic1cVM1ZvjFR1MiPLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoVLWMWUFM1YysFSwn2ZHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZHMGNUE1YqwVXn4BZic1cVM1ZvjFR2MiPLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoWLWc1cFM1cHIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIUXkUjQgoWRogjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogzb3TEVxEkLLglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOUmKUAkTEQ0TEEUUQIUSq8zM2HzTmkTLhkiKt3hKt3hKt3hKt3haTU0PUQDU3sFaicVTWkEQEYzXmEDTtDDRTQlcEEiX4EDTtDDSt3xXt3hKt3hKt3hKlIUUMQUTPkzUZESQFM1ZQQEV5UjQ77RRC8Vav8lak4Fc9vyKVMEUy.Ea0cVZtMEcgQWY9.."
						}
,
						"snapshotlist" : 						{
							"current_snapshot" : 0,
							"entries" : [ 								{
									"filetype" : "C74Snapshot",
									"version" : 2,
									"minorversion" : 0,
									"name" : "LABS",
									"origin" : "LABS.vst3",
									"type" : "VST3",
									"subtype" : "Instrument",
									"embed" : 1,
									"snapshot" : 									{
										"pluginname" : "LABS.vst3",
										"plugindisplayname" : "LABS",
										"pluginsavedname" : "/Volumes/C/Program",
										"pluginsaveduniqueid" : 0,
										"version" : 1,
										"isbank" : 0,
										"isbase64" : 1,
										"sliderorder" : [  ],
										"slidervisibility" : [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
										"blob" : "13241.VMjLg.6L...OVMEUy.Ea0cVZtMEcgQWY9vSRC8Vav8lak4Fc9jCN2TiKV0zQicUPt3hKl4hKt3BTt3hKt3hKLoGVzMGQt3BV3QlQIoGTtEjKt3BR1QEa2oFVtPDTAUkKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKtrFSzU0PIMERjYkPt3hc48zLvXTXlg0UYgWSWoUczX0SnQTZKYGRBgzZzDCV0EkUZQ2XV8DZTUTUFAiPNg1Mo8jY1MzTmkTLhkicSMUQQUETlgkUXM2ZFEFMvjFRDkzUiMWSsgjYyXEVyUkUOgFTpIFLvDiXn4hPhgGNFkELMYzXMgiQYsFLogjcHIDRwTEahk2ZwDFcvjFR2MiPLQGSogjYPcEVs0zUOgFQCwzctLDS14RZNQTRWM1bM0FRloWLgo1Zrk0aUYTV3fjPLg1Mn8zMTUkTlQ0UZk2ZrQ1ZvjFR2MiPLglKRM1aMESXxcmUXYWSWkkZvjFR2gDdKkicSAkTQUkTC0zZOcCSUEEUQUkTNMFQH8VTV8DZtHyU4sVagkVTvDFUUYUX1gCaHYFVWgkbUcUV3fjTLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWo1ZsE1YvXkVo0TaUs1cwDVZqYzXz.idgoVUrgjYXcEVxU0UYgCR3A0SvPDURUkdTMUUDEkYXUUTLgidPkTTUYkYlQkTGclZHU2LC8zTUQTUTslZScTPRokZvjFR1gCLh4FNrIldIUTUMgiQYsVRBgTLEYTXvTkUOgldnwDctjFR0MyPOMUUDUEUqo1TGEjTZoFLogjc3r1XqcWLgk1ZFMFMMQ0X3k0UYglKnM1Y2Y0XqASZHwzZpMUQEoFUlgUUQwDN5AURQUkUncCZOcCSUEEUQUkTNMFQH8VTV8DZtHyU2U0UXQWTWoUdUY0T0EkUYglKnM1Y2Y0XqASZHg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWM2ZFQ1ZIcETpk0UXQWSVkkZIIDRwTjQgASUV8DZDkFR0MyPOMUUDUEUqo1TGEjTZoFLogjc3TUXuc1UYg2XDEVcIYEVxkjPHESQFEFLUY0SnQTZHU2LC8zTUQTUTslZScTPRokZvjFR1gSUg81YWkEd2oWXoMGaHYFVWgkbUcUV3fjPLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWM2ZFQ1ZIcDUmMlUYglKnM1Y2Y0XqASZHg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWcVRGM1aMYzT00TLZglKnM1Y2Y0XqASZHYGR3sTN1kGUEEUQUkDM5EkYpYTV3fjPhUVQrIldqECVPUTLYsVRBgTLEYTXvTkUOgFR3sTN1kGUEEUQUkDM5EkYpYTV3fjPhU1cVgUdQICUqcmUYkVTWkkZAslXuAiUXg2ZWAEdQckVokjPHESQFEFLUY0Sn4RZHU2LC8zTUQTUTslZScTPRokZvjFR1gSUg8VTVo0PmYEVzQiUYIWRBgTLEYTXvTkUOglKogTcyLzSSUEQUQ0ZpM0QAIkVpASZHYGNvH1Z2YUVoE0UYoVTUgUaM0FRlg0UXIWUWkENHgFR0MyPOMUUDUEUqo1TGEjTZoFLogjc3rVVucmQisVRGQUcM0FRlg0UXIWUWkENHgmUikDdKkic4QUQQUTUIQidQYlZFkENHIjXkETahsVSWkkdAASX4kjPHESQFEFLUY0SnIWUWg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWsVVxD1P3vVX5UjUZQWUrIFT3DiXn4BZic1cVM1ZvjFRgAyZHU2LC8zTUQTUTslZScTPRokZvjFR1gCLgkVTWgULUwFRlg0UXIWUWkENHIDSncCZOcCSUEEUQUkTNMFQH8VTV8DZtHyU0EkUYkWTWIEcQYkVoUULhglKnM1Y2Y0XqASZHYmcRwjbHMzR4YmPMIGUCsTL1gWSxY1PKQicRwjc1IES2YmTLgmcRwTd1IES5YmTLAicRwTL1IESxXmTLMicRwDM1gFS1YGZLcmcnwDd1gFS4YGZLomcnwDL1gFSwXGZLIicnwzL1gFSzXGdLYmc3wzc1gGS3YGdLkmc3wjd1gGSvXGdLEic3wjL1gGSyXGdLQicB0jc1ITS2YmPMgmcB0Td1ITS5YmPMAicB0TL1ITSxXmPMMicB0DM1IUS1YmTMcmcR0Dd1IUS4YmTMomcR0DL1IUSwXmTMIicR0zL1IUSzXGZMYmcn0zc1gVS3YGZMkmcn0jd1gVSvXGZMEicn0jL1gVSyXGZMQic30jc1gWS2YGdMgmc30Td1gWS5YGdMAic30TL1gWSxXGdMMic30DM1IjS1YmPNcmcB4Dd1IjS4YmPNomcB4DL1IjSwXmPNIicB4zL1IjSzXmTNYmcR4zc1IkS3YmTNkmcR4jd1IkSvXmTNEicR4jL1IkSyXmTNQicRwjctLzR24xTLIGQCwDd1IES1wzPKcmKC0jbDMDSvXmTLYGVCszctjWSxQzPLMicRwjcpMzR2QzPLIGQSwzc1IES2gzPKcGQ4wjbDMES5YmTLcGUCszcDkVSxQzTLIicRwzclMzR2QzTNIGQowjc1IES3QzPKcGRowjbDkFS4YmTLgGTCszcHMUSxQTZLEicRwDdhMzR2gzPNIGQowDM1IES44xPKcGSSwjbDkGS3YmTLkGSCszcLMTSxQTdLAicRwTdXMzR2wTdMIGQ4wzL1IES4o1PKcGTCwjbDMTS2YmTLoGRCszcPkGSxQzPMomcRwjdTMzR2AUZMIGQC0jL1IES5Y1PKcGTS4jbDMUS1YmTLACQCszcTkFSxQzTMkmcRwDLPMzR2Q0TMIGQS0TL1IESvH1PKcGUC4jbDMUSzXmTLEiKCszcXMESxQTZMgmcRwTLLMzR2g0PMIGQo0DL1IESwf0PKcGV40jbDkVSyXmTLEiZCszchMDSxQTdMcmcRwjLHMzR2IVdLIGQ40jd1IESxP0PKcmXo0jbDkWSxXmTLIiYCszchMkSxQzPNYmcRwzLDMzR2YVZLIGQC4Td1IESy.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.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbEdUw1XqkTaXglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHcmZogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbkdqESVtEUaHYFVWgkbUcUV3fjPLQGUogjYLECV3fjTLMCRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNvHlcUYUVpkjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUVRWkkbUYEV4UEaHYFVWgkbUcUV3fjPLQGUogjYLECV3fjTLICRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUk0LA0lXq0jLh8FNrEFZtf1XmcmUisFLogzcyHDSn4BdXkFLogzcDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUpsVagcFLVoUZM0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHIESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkk0UXg2ZVgkdqESXzkjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SngzPLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUVVWoEZIcEV5gCaHYFVWgkbUcUV3fjTLQmKogjYLECV3fDZLcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUkEcYcEV5EUaHYFVWgkbUcUV3fjPLQmKS4DMpMkSzn1PNECUC4zLpMUS5oVdLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoVLWsFMrMlZUECVn4BZic1cVM1ZvjFR1MiPLQiZS4DMpMkSz.0PLkmZS0TdTMUS3gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkUEagESSWMVdIIDRwTjQgASUV8DZDk1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkUEagESRWkkbIIDRwTjQgASUV8DZHk1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkAiUZMSSWo0bAcTXqkjPHESQFEFLUY0Sn4RZKACRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fDdhUVVFE1aA0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFR4gCLi8VTFMlaIIDRwTjQgASUV8DZtj1RvfjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHgmXkEzUXQWRBgTLEYTXvTkUOglKosDLHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUYQYUVxUjUjglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwb0YvDSXvPiQiglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbkZqEiX5gCahoWRBgTLEYTXvTkUOglKosjcHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUY2Y0X4cFaUsVRsgEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU1UkQYECNFEFZtf1XmcmUisFLogjcyHUSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU1UkQYQzZsEFZtf1XmcmUisFLogjcyHUSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUxUEag0VTGoEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU5slUgsFLTgUZmYkVzUEaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNvHldIcUV50jQZglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwbUd3vVV5ETUYoVQFEFZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyU3UULhYGNrEVdUwFRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugSUgc1cFE1ZQ0FRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugCLhoGNFI1ZvP0X5UEaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNEk0aIcUVoE0UZUGMrgjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogza3TTX1kEUZIWTWkEdIIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkEjLggWTWg0bUwVX5gCaHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZH8FNUE1ZQcEVgUzPNMVRBgTLEYTXvTkUOglKosjcHIDRo0jUOgldRwDZtfGVokkZhUGLV8DZtj1R1gjPHkVSFUUcvjFR2MiPLglK3gUZMAiVqM1UOgFQosjcHIDRo0DaQI2ZFIFNHIDSncCZOciKUAkTEQ0TlolQYgCRRoUYvXUV5UTLVcmZScEZtf1XmcmUisFLogjcyHDSn4BdXkFLogzbDkFRlwTLXYTRxD1bvjFR1MiPLglK3gUZQASX3fjTLQmKogjYLECVSMmUYICLogzcyHDSn4BdXkVVDE1aAc0Sn4RZHU2LC8DTEoFUAACQH8VTV8DZpEyUyUkQic1bqwjc5sFRlg0UXIWUWkENHIDSz4RZHYFSwfENHI0R2gjPHkVSrEEd3XUX3fjPLQmKogjYLECVTgiUOgFQosjcHIDRo0TLTEWUwLFNHIESz4RZHYFSwfkQ2YkV1ASZHYGR3sTN1MDUAkTUP0TPRokZvjFRugSUgsVTWgUXIMESikjPHESQFEFLUY0Sn4RZKYGRBgTZMY0SnomTLglK3gUZYolX0AiUOglKosjcHIDRo0jQUUGLogzcyHDSn4BdXkVSvn0Zic0SnQTZKYGRBgTZMwVTxslQhgCRBwDZ2f1S23RUPIUQTMkYpYTV3fjTZUFLVkkdEEiU3gzTWglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOAUQpQUPvPDRuEkUOglZwb0bUYzXmM2ZLkmdqgjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogza3TUXqE0UXEVRC0zXIIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIkVkAiUYoWQwXEdTM0Un4BZic1cVM1ZvjFR1MiPLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoVLWMWUFM1YysFSwn2ZHYFVWgkbUcUV3fjPLQmKogjYLECV3fjTKcGRBgTZMwVT3giUggCRBwDctjFRlwTLXQENV8DZDk1R1gjPHkVSwPUbUEyX3fjTLQmKogjYLECVFcmUZYGLogjcHg2R4X2PTETRUAUSAIkVpASZHMGNUE1YqwVXn4BZic1cVM1ZvjFR2MiPLglK3gUZvjFRyQTZHYFSwfkQIISXyASZHY2LBwDZtfGVoEELggCRRwDctjFRlwTLXM0bVkkLvjFR2MiPLglK3gUZYQTXuEzUOglKogTcyLzSPUjZTEDLDgzaQY0SnoWLWc1cFM1cHIDRwTjQgASUV8DZtj1R1gjPHkVSV8DZ5IESn4BdXkVVpIVcvX0Sn4RZKYGRBgTZMYTU0ASZHc2LBwDZtfGVo0DLZs1XW8DZDk1R1gjPHkVSrEkbqYjX3fjPLg1Mn8zMtTETRUDUSYlZFkENHIUXkUjQgoWRogjYXcEVxU0UYgCRBwDctjFRlwTLXgCRRszcHIDRo0DaQgGNVEFNHIDSz4RZHYFSwfEU3X0SnQTZKYGRBgTZMECUwUULigCRRwDctjFRlwTLXYzcVokcvjFR1gDdKkicCQUPIUETMEjTZoFLogzb3TEVxEkLLglKnM1Y2Y0XqASZHY2LBwDZtfGVoASZHMGQogjYLECVFkjLgMGLogjcyHDSn4BdXkVTvDFNHIESz4RZHYFSwf0TyYUVx.SZHc2LBwDZtfGVokEQg8VPW8DZtjFR0MyPOUmKUAkTEQ0TEEUUQIUSq8zM2HzTmkTLhkiKt3hKt3hKt3hKt3haTU0PUQDU3sFaicVTWkEQEYzXmEDTtDDRTQlcEEiX4EDTtDDSt3xXt3hKt3hKt3hKlIUUMQUTPkzUZESQFM1ZQQEV5UjQ77RRC8Vav8lak4Fc9vyKVMEUy.Ea0cVZtMEcgQWY9.."
									}
,
									"fileref" : 									{
										"name" : "LABS",
										"filename" : "LABS.maxsnap",
										"filepath" : "~/Documents/Max 8/Snapshots",
										"filepos" : -1,
										"snapshotfileid" : "db66b03da6dd1cbe7b90556a8849a5c2"
									}

								}
 ]
						}

					}
,
					"text" : "vst~ \"C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3\"",
					"varname" : "vst~",
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-103",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 347.0, 418.166667999999959, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-101",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "bang", "bang" ],
					"patching_rect" : [ 347.0, 386.666667999999959, 42.0, 22.0 ],
					"text" : "edge~"
				}

			}
, 			{
				"box" : 				{
					"channels" : 1,
					"id" : "obj-100",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 1,
					"numoutlets" : 4,
					"orientation" : 1,
					"outlettype" : [ "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 472.0, 357.666667999999959, 136.0, 41.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_longname" : "live.gain~[2]",
							"parameter_mmax" : 6.0,
							"parameter_mmin" : -70.0,
							"parameter_shortname" : "guitar",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "live.gain~[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-98",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 445.0, 411.5, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-55",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 17.5, 318.0, 32.0, 22.0 ],
					"text" : "train"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 13.0,
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "" ],
					"patching_rect" : [ 160.0, 268.999995589256287, 90.0, 23.0 ],
					"text" : "getattr latency"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 13.0,
					"id" : "obj-17",
					"linecount" : 2,
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "" ],
					"patching_rect" : [ 195.5, 314.0, 174.5, 38.0 ],
					"text" : "fluid.onsetslice~ @metric 9 @threshold 0.22"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-52",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 17.5, 242.333323178512615, 33.0, 22.0 ],
					"text" : "read"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 17.5, 268.999995589256287, 32.0, 22.0 ],
					"text" : "start"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 11,
					"outlettype" : [ "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "", "" ],
					"patching_rect" : [ 17.5, 434.5, 124.0, 22.0 ],
					"text" : "rolypoly~ tim.mid"
				}

			}
, 			{
				"box" : 				{
					"attr" : "latency",
					"id" : "obj-57",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 160.0, 386.666667999999959, 150.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"attr" : "filter_hits",
					"id" : "obj-4",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 67.5, 212.999995589256287, 225.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"attr" : "out_message",
					"id" : "obj-5",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 67.5, 186.999995589256258, 150.0, 22.0 ]
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 0 ],
					"source" : [ "obj-1", 9 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 1 ],
					"order" : 0,
					"source" : [ "obj-100", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 1,
					"source" : [ "obj-100", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-103", 0 ],
					"source" : [ "obj-101", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-11", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 1 ],
					"source" : [ "obj-12", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"source" : [ "obj-13", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-57", 0 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.701961, 0.701961, 0.701961, 0.0 ],
					"destination" : [ "obj-20", 0 ],
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"order" : 1,
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-101", 0 ],
					"order" : 0,
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 1 ],
					"midpoints" : [ 209.5, 517.567627000000016, 135.0, 517.567627000000016 ],
					"source" : [ "obj-18", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"midpoints" : [ 121.5, 516.567627000000016, 121.5, 516.567627000000016 ],
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-100", 0 ],
					"order" : 0,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"order" : 1,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.701961, 0.701961, 0.701961, 0.0 ],
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 1 ],
					"source" : [ "obj-30", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-31", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 77.0, 419.0, 27.0, 419.0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 77.0, 209.0, 77.0, 209.0, 77.0, 419.0, 27.0, 419.0 ],
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-52", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-55", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 169.5, 419.0, 27.0, 419.0 ],
					"source" : [ "obj-57", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-100" : [ "live.gain~[2]", "guitar", 0 ],
			"obj-12" : [ "vst~", "vst~", 0 ],
			"obj-2::obj-21::obj-6" : [ "live.tab[3]", "live.tab[1]", 0 ],
			"obj-2::obj-35" : [ "[5]", "Level", 0 ],
			"obj-30" : [ "live.gain~[3]", "drums", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "LABS.maxsnap",
				"bootpath" : "~/Documents/Max 8/Snapshots",
				"patcherrelativepath" : "../../../Max 8/Snapshots",
				"type" : "mx@s",
				"implicit" : 1
			}
, 			{
				"name" : "demosound.maxpat",
				"bootpath" : "C74:/help/msp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "fluid.onsetslice~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "interfacecolor.js",
				"bootpath" : "C74:/interfaces",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "random.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "roly.browse.maxpat",
				"bootpath" : "~/Documents/GitHub/rolypoly/patchers",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rolypoly~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "saw.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "sine.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "square.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
